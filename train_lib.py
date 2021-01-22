import tensorflow as tf
import collections
import time
import datetime
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import io
from audio_feature_reconstruction.gan import data_provider
from audio_feature_reconstruction.gan import networks

HParams = collections.namedtuple('HParams', [
    'embedding_generator',
    'tfrecord_path_train',
    'tfrecord_path_val',
    'audio_tracks_path',
    'saved_model_path',
    'examples_per_record',
    'batch_size',
    'buffer_size',
    'layer_name',
    'train_log_dir',
    'generator_lr',
    'discriminator_lr',
    'adv_loss_weight',
    'l1_loss_weight',
    'epochs',
    'ps_replicas',
])

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def combined_generator_loss(fake_output, fake_images, real_images, l1_loss_weight, gan_loss_weight):
  gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
  l1_loss = tf.reduce_mean(tf.abs(fake_images - real_images))
  total_gen_loss = gan_loss_weight * gan_loss + l1_loss_weight * l1_loss
  return total_gen_loss, gan_loss, l1_loss


def read_track_id(tf_example):
    track_id = tf_example.features.feature['track_id'].bytes_list.value[0].decode("utf-8")
    return track_id


def read_spectrogram(tf_example):
    chunked_spectrogram = tf_example.features.feature['chunked_spectrogram'].float_list.value
    n_chunks = tf_example.features.feature['chunked_spectrogram_shape'].int64_list.value[0]
    n_frames = tf_example.features.feature['chunked_spectrogram_shape'].int64_list.value[1]
    n_bands = tf_example.features.feature['chunked_spectrogram_shape'].int64_list.value[2]
    chunked_spectrogram = np.reshape(chunked_spectrogram,(n_chunks,n_frames,n_bands))
    return chunked_spectrogram, n_chunks, n_frames, n_bands


def read_embedding(tf_example):
    # Read embeddings
    embeddings = tf_example.features.feature['module_apply_default/embedding'].float_list.value
    n_embeddings = tf_example.features.feature['module_apply_default/embedding_shape'].int64_list.value[0]
    embedding_dim = tf_example.features.feature['module_apply_default/embedding_shape'].int64_list.value

    embeddings = np.reshape(embeddings, embedding_dim)
    return embeddings, n_embeddings, embedding_dim


def read_tfrecord(tfrecord_paths, tfrecord_idx):
  chunked_spectrogram_dict = {}
  embeddings_dict = {}

  for example in tf.compat.v1.python_io.tf_record_iterator(tfrecord_paths[tfrecord_idx]):
    tf_example = tf.train.Example.FromString(example)

    track_id = read_track_id(tf_example)

    chunked_spectrogram, n_chunks, n_frames, n_bands = read_spectrogram(tf_example)
    chunked_spectrogram = np.reshape(chunked_spectrogram, (n_chunks * n_frames, n_bands)).transpose()
    embeddings, n_embeddings, embedding_dim = read_embedding(tf_example)

    # Save value in dictionary
    chunked_spectrogram_dict[track_id] = chunked_spectrogram
    embeddings_dict[track_id] = embeddings

  return embeddings_dict, chunked_spectrogram_dict


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def generate_and_save_images(hparams, model, epoch, summary_writer, discriminator, dataset_val):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    tfrecord_paths = tf.io.gfile.glob(hparams.tfrecord_path_val + '*.tfrecord')
    tfrecord_idx = 32
    embeddings_dict, chunked_spectrogram_dict = read_tfrecord(tfrecord_paths, tfrecord_idx)
    # Extract audio track names contained in tfrecord
    tfrecord_track_names = list(chunked_spectrogram_dict.keys())

    # Randomly select one track
    track_idx = random.randrange(len(tfrecord_track_names))

    # Extract embeddings
    embeddings_temp = embeddings_dict[tfrecord_track_names[track_idx]]

    # Number of chunks in embeddings (may vary) N.B. is the same for the spectrogram
    n_chunks = embeddings_temp.shape[0]

    # Read corresponding spectrograms
    chunked_spectrogram = chunked_spectrogram_dict[tfrecord_track_names[track_idx]]

    estimated_spectrogram = model(embeddings_temp, training=False)
    n_frames = 96
    n_bands = 64
    estimated_spectrogram = np.reshape(estimated_spectrogram, (n_frames * n_chunks, n_bands)).transpose()

    figure = plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1, title='Ground Truth')
    plt.ylabel('Bands')
    plt.imshow(chunked_spectrogram, aspect='auto')
    plt.subplot(2, 1, 2, title='Estimated Epoch' + str(epoch))
    plt.imshow(estimated_spectrogram, aspect='auto')
    plt.xlabel('Frames')
    plt.ylabel('Bands')

    with summary_writer.as_default():
        # Val image
        tf.summary.image("Training data", plot_to_image(figure), step=epoch)

    gen_loss_val = []
    gan_loss_val = []
    l1_loss_val = []
    disc_loss_val = []

    # Compute losses
    for embedding, images in dataset_val:
      generated_images=model(embedding, training=False)
      real_output = discriminator(images, training=False)
      fake_output = discriminator(generated_images, training=False)
      disc_loss_val_temp = discriminator_loss(real_output, fake_output)
      gen_loss_val_temp, gan_loss_val_temp, l1_loss_val_temp = combined_generator_loss(
        fake_output, generated_images, images, hparams.l1_loss_weight, hparams.adv_loss_weight)

    gen_loss_val.append(gen_loss_val_temp)
    gan_loss_val.append(gan_loss_val_temp)
    l1_loss_val.append(l1_loss_val_temp)
    disc_loss_val.append(disc_loss_val_temp)

    with summary_writer.as_default():
        # Val loss
        tf.summary.scalar('gen_total_loss_validation', np.mean(gen_loss_val), step=epoch)
        tf.summary.scalar('gen_gan_loss_validation', np.mean(gan_loss_val), step=epoch)
        tf.summary.scalar('gen_l1_loss_validation', np.mean(l1_loss_val), step=epoch)
        tf.summary.scalar('disc_loss_validation', np.mean(disc_loss_val), step=epoch)


def train(hparams):
    # Number of epochs
    epochs = hparams.epochs

    # Load training data
    datapath = tf.io.gfile.glob(hparams.tfrecord_path_train+'*.tfrecord')
    dataset = data_provider.create_dataset(
        datapath[:32], hparams.buffer_size, hparams.batch_size,hparams)

    dataset_val = data_provider.create_dataset(
      datapath[32], hparams.buffer_size, hparams.batch_size, hparams)

    # Load generator model
    generator = networks.make_generator_model(hparams)

    # Load discriminator model
    discriminator = networks.make_discriminator_model()

    generator.summary()
    discriminator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(hparams.generator_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(hparams.discriminator_lr)

    # String used to identify current model training
    train_instance = '_'+hparams.embedding_generator+'_'+hparams.layer_name+'_l1_'+str(hparams.l1_loss_weight)+'_adv_'+str(hparams.adv_loss_weight)

    summary_writer = tf.summary.create_file_writer(
        hparams.train_log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + train_instance+'_'+str(hparams.epochs)+'_epochs')

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(noise, images, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss, gan_loss, l1_loss = combined_generator_loss(fake_output,
                                                                  generated_images, images,
                                                                  hparams.l1_loss_weight,
                                                                  hparams.adv_loss_weight)

            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

        return gen_loss, gan_loss, l1_loss, disc_loss

    # TRAINING
    for epoch in range(hparams.epochs):
        start = time.time()

        for chunked_spectrogram, embedding in dataset:
            gen_loss, gan_loss, l1_loss, disc_loss = train_step(chunked_spectrogram, embedding, epoch)

        # Produce images for the GIF as we go

        if epoch % 10 == 0:
            generate_and_save_images(hparams, generator,
                                     epoch + 1,
                                     summary_writer, discriminator, dataset_val)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            print('epoch ' + str(epoch))
            print('gen_loss: ' + str(gen_loss))
            print('gan_loss:' + str(gan_loss))
            print('l1_loss:' + str(l1_loss))
            print('disc_loss: ' + str(disc_loss))

    # Generate after the final epoch
    if hparams.adv_loss_weight == 0:
      loss_type = 'l1'
    elif hparams.l1_loss_weight == 0:
      loss_type = 'adv'
    else:
      loss_type = 'l1_adv'

    saved_model_path = os.path.join(hparams.saved_model_path,hparams.embedding_generator + '_' +hparams.layer_name, loss_type)

    generator.save(saved_model_path)



