# Copyright 2020 name of copyright owner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

import tensorflow as tf

"""
Data input pipeline
"""

# Shape of embedding layers
layers_shape = {'pool1': [None, 48, 32, 64],
                'pool2': [None, 24, 16, 128],
                'pool3': [None, 12, 8, 256],
                'pool4': [None, 6, 4, 512],
                'fc1_1': [None, 4096],
                'fc1_2': [None, 4096],
                'fc2': [None, 128],
                'features': [None, 96, 64],
              }


def create_dataset(filepath, buffer_size, batch_size, hparams):
  """Applies `_parse(serialized_example)` function to TFRecordDataset
  Args:
    filepath: String. Path to TFRecords Folder containing the dataset
    buffer_size: Integer. Number of element from this dataset from which the new dataset will sample.
    batch_size: Integer. Number of consecutive elements of this dataset to combine in a single batch
  Returns:
    `embedding_reshape` and `chunked_spectrogram_reshape`  batches of Tensors corresponding to embeddings and
    log-mel spectrograms extracted and parsed randomly from `dataset`
  """

  def _parse(serialized_example):
    """Parses a tensorflow.SequenceExample into randomly extracted embedding vector and log-mel spectrogram chunk

    Args:
      serialized_example: Tensor. Single serialized SequenceExample
    Returns:
      `embedding_reshape` and `chunked_spectrogram_reshape` One-element batch of Tensors corresponding to embedding and
       log-mel spectrogram extracted and parsed randomly from `serialized_example`

    """
    feature_description = {
      'Sigmoid_shape': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
      'chunked_spectrogram': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
      'module_apply_default/embedding': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,
                                                                      allow_missing=True),
      'chunked_spectrogram_shape': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
      'audio_samples': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
      'label_mask': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
      'spectrogram_shape': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
      'Sigmoid': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
      'module_apply_default/embedding_shape': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,
                                                                            allow_missing=True),
      'labels': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
      'spectrogram': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    # Load data
    embedding = example['module_apply_default/embedding']
    chunked_spectrogram = example['chunked_spectrogram']

    # Load shapes
    embedding_shape = example['module_apply_default/embedding_shape']
    chunked_spectrogram_shape = example['chunked_spectrogram_shape']

    # Reshape to original data shapes

    # Embeddings -> n_slices x n_frames x n_bands
    embedding_reshape = tf.reshape(embedding, embedding_shape)

    # Chunked spectrogram -> n_slices x embedding_dim
    chunked_spectrogram_reshape = tf.reshape(chunked_spectrogram, chunked_spectrogram_shape)

    # Select random idx to extract one pair of (embeddings,chunked_spectrogram)
    batch_idx = tf.random.uniform(shape=(),
                                  minval=0,
                                  maxval=embedding_shape[0],
                                  dtype=tf.int64
                                  )

    # Chunked spectrogram
    chunked_spectrogram_reshape = tf.slice(chunked_spectrogram_reshape,
                                           begin=[batch_idx, 0, 0],
                                           size=[1, chunked_spectrogram_shape[1], chunked_spectrogram_shape[2]]
                                           )

    # Chunked spectrogram needs to comply with the channel last format required by the tf.keras model
    chunked_spectrogram = tf.expand_dims(chunked_spectrogram_reshape, axis=3)

    # Embeddings
    # N.B. depending on the selected layer the slice operation changes
    # Tensorflow does not know shape of the tensors at graph generation time

    # 4D tensors
    if (hparams.layer_name == 'pool1' or hparams.layer_name == 'pool2' or
            hparams.layer_name == 'pool3' or hparams.layer_name == 'pool4'):
      embedding = tf.slice(embedding_reshape,
                           begin=[batch_idx, 0, 0, 0],
                           size=[1, embedding_shape[1], embedding_shape[2], embedding_shape[3]]
                           )
    # 2D tensors
    else:
      embedding = tf.slice(embedding_reshape,
                           begin=[batch_idx, 0],
                           size=[1, embedding_shape[1]]
                           )
    # Embedding must be squeezed along axis one to be in the required format
    embedding = tf.squeeze(embedding, axis=0)
    chunked_spectrogram = tf.squeeze(chunked_spectrogram, axis=0)

    # Rough spectrogram normalization
    normalizing_factor = 10  # Normalization factor applied to log-mel spectrogram
    chunked_spectrogram = tf.divide(chunked_spectrogram, normalizing_factor)

    return embedding, chunked_spectrogram

  # Create dataset
  dataset = tf.data.TFRecordDataset(filepath)

  # Map parser on every filepath in the array.
  dataset = dataset.map(_parse)  # check num_parallel_calls

  # Set the batch size
  dataset = dataset.shuffle(buffer_size)

  # Set number of datapoints to load and shuffle
  dataset = dataset.batch(batch_size)

  return dataset


