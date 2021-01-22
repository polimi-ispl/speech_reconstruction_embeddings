import argparse
import os
import numpy as np
import tensorflow as tf
from tqdm import trange
from params import logmel_root, emb_root, tf_root
import glob

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():
    # Arguments parser
    parser = argparse.ArgumentParser(description='Extract embeddings.')
    parser.add_argument('--set_name', type=str, help='Dataset name', default='custom_set')
    parser.add_argument('--audio_dir', type=str, help='Folder containing audio tracks', default='.')
    parser.add_argument('--audio_format', type=str, help='Audio format extension', default='wav')
    parser.add_argument('--model_name', type=str, help='Model name', default='VGGish')
    parser.add_argument('--layer', type=str, help='Layer for feature extraction', default='')
    parser.add_argument('--n_songs', type=int, help='Number of songs', default=100)
    parser.add_argument('--group', type=int, help='Number of songs per TFRecord', default=100)
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

    # Output folders
    if not os.path.isdir(tf_dir):
        os.makedirs(tf_dir)

    # Generate data list
    audio_path_list = glob.glob(os.path.join(audio_dir, '*.{:s}').format(audio_format))
    logmel_path_list = glob.glob(os.path.join(logmel_dir, '*.npy'))
    emb_path_list = glob.glob(os.path.join(emb_dir, '*.npy'))

    # Sort
    audio_path_list.sort()
    logmel_path_list.sort()
    emb_path_list.sort()

    # Select number of audio
    if n_songs > 0:
        audio_path_list = audio_path_list[0:n_songs]
        logmel_path_list = logmel_path_list[0:n_songs]
        emb_path_list = emb_path_list[0:n_songs]

    # Loop over data
    group_cnt = 0
    tracks_cnt = 0
    for i in trange(len(audio_path_list)):

        # Paths
        audio_path = audio_path_list[i]
        logmel_path = logmel_path_list[i]
        emb_path = emb_path_list[i]

        # Output path
        id = os.path.basename(logmel_path).split('.')[0]
        if not np.mod(tracks_cnt, group):
            tracks_cnt = 0
            tf_path = os.path.join(tf_dir, '{:010d}.tfrecord'.format(group_cnt))
            writer = tf.io.TFRecordWriter(tf_path)

        try:
            # Load data
            logmel = np.load(logmel_path)
            emb = np.load(emb_path)

            # Create example
            feature = {
                'track_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                'chunked_spectrogram': tf.train.Feature(float_list=tf.train.FloatList(value=logmel.flatten())),
                'chunked_spectrogram_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=logmel.shape)),
                'module_apply_default/embedding': tf.train.Feature(float_list=tf.train.FloatList(value=emb.flatten())),
                'module_apply_default/embedding_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=emb.shape))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Create and save TFrecord
            writer.write(example.SerializeToString())

            # Adjust counters
            if tracks_cnt == 0:
                group_cnt += 1
            tracks_cnt += 1

        except:
            pass
            #print('Cannot process file {:s}'.format(id))


if __name__ == '__main__':
    main()
