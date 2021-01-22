import argparse
import os
import numpy as np
import tensorflow as tf
from params import logmel_root, emb_root, tf_root
import glob
from functools import partial

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    # Arguments parser
    parser = argparse.ArgumentParser(description='Extract embeddings.')
    parser.add_argument('--set_name', type=str, help='Dataset name', default='custom_set')
    parser.add_argument('--audio_dir', type=str, help='Folder containing audio tracks', default='.')
    parser.add_argument('--audio_format', type=str, help='Audio format extension', default='wav')
    parser.add_argument('--model_name', type=str, help='Model name', default='VGGish')
    parser.add_argument('--layer', type=str, help='Layer for feature extraction', default='')
    parser.add_argument('--n_songs', type=int, help='Number of songs', default=100)
    parser.add_argument('--group', type=int, help='Number of songs per TFRecord', default=400)

    args = parser.parse_args()
    set_name = args.set_name
    audio_dir = args.audio_dir
    audio_format = args.audio_format
    model_name = args.model_name
    layer = args.layer
    n_songs = args.n_songs
    group = args.group

    # Folders
    logmel_dir = os.path.join(logmel_root, set_name)
    emb_dir = os.path.join(emb_root, set_name, model_name + '_' + layer)
    tf_dir = os.path.join(tf_root, set_name, model_name + '_' + layer, str(group))

    # Generate data list
    audio_path_list = glob.glob(os.path.join(audio_dir, '*.{:s}').format(audio_format))
    logmel_path_list = glob.glob(os.path.join(logmel_dir, '*.npy'))
    emb_path_list = glob.glob(os.path.join(emb_dir, '*.npy'))
    tf_path_list = glob.glob(os.path.join(tf_dir, '*.tfrecord'))

    # Sort
    audio_path_list.sort()
    logmel_path_list.sort()
    emb_path_list.sort()
    tf_path_list.sort()

    # Select number of audio
    if n_songs > 0:
        audio_path_list = audio_path_list[0:n_songs]
        logmel_path_list = logmel_path_list[0:n_songs]
        emb_path_list = emb_path_list[0:n_songs]

    # File name lists
    audio_name_list = [os.path.basename(x).split('.')[0] for x in audio_path_list]
    logmel_name_list = [os.path.basename(x).split('.')[0] for x in logmel_path_list]
    emb_name_list = [os.path.basename(x).split('.')[0] for x in emb_path_list]

    # Generate feature example
    logmel = np.load(logmel_path_list[0])
    emb = np.load(emb_path_list[0])
    feature = {
        'track_id': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'chunked_spectrogram': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'chunked_spectrogram_shape': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'module_apply_default/embedding': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'module_apply_default/embedding_shape': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    parse_function = lambda example, feature : tf.io.parse_single_example(example, feature)
    parse_function_par = partial(parse_function, feature=feature)

    tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),

    # Loop over data
    tf_name_list = []
    for tf_path in tf_path_list:
        raw_dataset = tf.data.TFRecordDataset(tf_path)
        parsed_dataset = raw_dataset.map(parse_function_par)
        for data in parsed_dataset:
            tf_name_list.append(data['track_id'].numpy()[0].decode("utf-8"))

    # Check for missing files
    missing_logmel = np.setdiff1d(audio_name_list, logmel_name_list)
    missing_emb = np.setdiff1d(audio_name_list, emb_name_list)
    missing_tf = np.setdiff1d(audio_name_list, tf_name_list)

    # Print report
    print('Dataset: {} - Model: {} - Layer: {}'.format(set_name, model_name, layer))
    if len(missing_logmel) == 0 & len(missing_emb) == 0 & len(missing_tf) == 0:
        print('All test passed.')
    else:
        if len(missing_logmel) > 0:
            print('  1. Missing logmel: {}'.format(missing_logmel))
        else:
            print('  1. Logmel test passed')
        if len(missing_emb) > 0:
            print('  2. Missing embedding: {}'.format(missing_emb))
        else:
            print('  2. Embedding test passed')
        if len(missing_tf) > 0:
            print('  3. TFRecords embedding: {}'.format(missing_tf))
        else:
            print('  3. TFRecords test passed')
    print('')


if __name__ == '__main__':
    main()
