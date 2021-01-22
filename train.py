# Lint as: python3
"""Training script"""

import os
from absl import flags
from absl import app
from audio_feature_reconstruction.gan import train_lib

flags.DEFINE_string('embedding_generator', 'VGGish', 'Architecture used to generate input embeddings')

flags.DEFINE_string('tfrecord_path_train', '/nas/public/exchange/audio_feature_reconstruction/tfrecords/lj_speech',
                    'Path  Containing tfrecords used for training')

flags.DEFINE_string('tfrecord_path_val', '/nas/public/exchange/audio_feature_reconstruction/tfrecords/lj_speech',
                    'Path  Containing tfrecords used for validation/testing')

flags.DEFINE_string('audio_tracks_path', '/nas/home/pbestagini/audioset_download/data/audio/lj_speech/',
                    'Path  Containing audio tracks corresponding to validation examples')

flags.DEFINE_string('saved_model_path', '/nas/public/exchange/audio_feature_reconstruction/models',
                    'Path  where to save corresponding estimator in SavedModel format')

flags.DEFINE_integer('examples_per_record', 400,'Number of examples contained in each tfrecord')

flags.DEFINE_integer('batch_size', 256, 'The number of embedding/spectrogram couple in each batch.')

flags.DEFINE_integer('buffer_size', 500, 'Size of buffer from which elements are randomly sampled.')

flags.DEFINE_string('layer_name', 'fc1_1', 'The name of the VGGish layer to feed into the network. Possible choices:'
                                                'pool1, pool2, pool3, pool4, fc1_1,fc1_2 embeddings.')

flags.DEFINE_string('train_log_dir', '/nas/home/lcomanducci/audio_feature_reconstruction/gan/logs',
                    'Directory where to write event logs.')

flags.DEFINE_float('generator_lr', 0.0001,
                   'The compression model learning rate.')

flags.DEFINE_float('discriminator_lr', 0.0001,
                   'The discriminator learning rate.')

flags.DEFINE_float('adv_loss_weight', 1.,
                   'Weight of the adversarial loss')

flags.DEFINE_float('l1_loss_weight', 100.,
                   'Weight of the l1 norm loss')

flags.DEFINE_integer('epochs', 300,
                     'The maximum number of gradient steps.')

flags.DEFINE_string('gpu', '0',
                     'Index of select gpu on the current machine.')

flags.DEFINE_integer(
    'ps_replicas', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

FLAGS = flags.FLAGS


def main(_):

  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
  os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

  hparams = train_lib.HParams(
    FLAGS.embedding_generator,
    '/nas/public/exchange/audio_feature_reconstruction/tfrecords/lj_speech/' + FLAGS.embedding_generator + '_' + FLAGS.layer_name + '/400/',
    '/nas/public/exchange/audio_feature_reconstruction/tfrecords/lj_speech/' + FLAGS.embedding_generator + '_' + FLAGS.layer_name + '/400/',
    FLAGS.audio_tracks_path,
    FLAGS.saved_model_path, FLAGS.examples_per_record, FLAGS.batch_size,
    FLAGS.buffer_size,  FLAGS.layer_name, FLAGS.train_log_dir,
    FLAGS.generator_lr, FLAGS.discriminator_lr, FLAGS.adv_loss_weight,
    FLAGS.l1_loss_weight, FLAGS.epochs, FLAGS.ps_replicas)

  train_lib.train(hparams)


if __name__ == '__main__':
  app.run(main)



