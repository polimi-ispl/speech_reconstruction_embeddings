import tensorflow as tf

from audio_feature_reconstruction.gan import data_provider


def make_generator_model(hparams):
  factor = 4
  initializer = tf.keras.initializers.truncated_normal(
    mean=0.0, stddev=0.01)

  model = tf.keras.Sequential()

  model.add(tf.keras.Input(shape=data_provider.layers_shape[hparams.layer_name][1:]))

  if hparams.layer_name == 'fc2'
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))

  if (hparams.layer_name == 'fc2'
    or hparams.layer_name  == 'fc1_2'):
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

  if hparams.layer_name == 'fc2' or hparams.layer_name == 'fc1_1':
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Reshaping to feed into network
    model.add(tf.keras.layers.Dense(6 * 4 * int(512 / factor)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Reshape((6, 4, int(512 / factor))))

  if (hparams.layer_name == 'fc2' or hparams.layer_name == 'fc1_2'
      or hparams.layer_name == 'fc1_1'or hparams.layer_name == 'pool4'):
    model.add(
      tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(
      tf.keras.layers.Conv2D(int(512 / factor), (3, 3), strides=(1, 1),
                    padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(
      tf.keras.layers.Conv2D(int(512 / factor), (3, 3), strides=(1, 1),
                    padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

  if (hparams.layer_name == 'fc2'
      or hparams.layer_name == 'fc1_2' or hparams.layer_name == 'fc1_1'
      or hparams.layer_name == 'pool4' or hparams.layer_name == 'pool3'):
    model.add(
      tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(
      tf.keras.layers.Conv2D(int(256 / factor), (3, 3), strides=(1, 1),
                    padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(
      tf.keras.layers.Conv2D(int(256 / factor), (3, 3), strides=(1, 1),
                    padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

  if (hparams.layer_name == 'fc2' or hparams.layer_name == 'fc1_2'
      or hparams.layer_name == 'fc1_1' or hparams.layer_name == 'pool4'
      or hparams.layer_name == 'pool3' or hparams.layer_name == 'pool2'
  ):
    model.add(
      tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(
      tf.keras.layers.Conv2D(int(128 / factor), (3, 3), strides=(1, 1),
                    padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

  if (hparams.layer_name == 'fc2' or hparams.layer_name == 'fc1_2'
      or hparams.layer_name == 'pool4' or hparams.layer_name == 'pool3'
      or hparams.layer_name == 'pool2' or hparams.layer_name == 'pool1'
  ):
    model.add(
      tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(
      tf.keras.layers.Conv2D(64/factor, (3, 3), strides=(1, 1),
                             padding='same', use_bias=False, kernel_initializer=initializer, activation='relu'))
    model.add(
      tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1),
                    padding='same', use_bias=False, kernel_initializer=initializer, activation='tanh'))

  assert model.output_shape == (None, 96, 64, 1)

  return model


def make_discriminator_model():
  factor = 4
  initializer = tf.keras.initializers.truncated_normal(
    mean=0.0, stddev=0.01)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(int(64 / factor), (4, 4), strides=(2, 2),
                          padding='same', kernel_initializer=initializer,
                          input_shape=[96, 64, 1], ))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.Conv2D(int(128 / factor), (4, 4), strides=(2, 2),
                          padding='same', kernel_initializer=initializer))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.Conv2D(int(256 / factor), (4, 4), strides=(2, 2),
                          padding='same', kernel_initializer=initializer))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.Conv2D(int(512 / factor), (4, 4), strides=(2, 2),
                          padding='same', kernel_initializer=initializer))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.Conv2D(1, 4, strides=1,
                                   padding='same', kernel_initializer=initializer))
  return model
