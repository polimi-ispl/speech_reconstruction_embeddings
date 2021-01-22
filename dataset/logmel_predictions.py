import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from params import logmel_predictions_root, emb_root, saved_models_root


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    # Arguments parser
    parser = argparse.ArgumentParser(description='Extract Log-mel spectrograms predictions.')
    parser.add_argument('--set_name', type=str, help='Dataset name', default='lj_speech')
    parser.add_argument('--model_name', type=str, help='Model name', default='VGGish')
    parser.add_argument('--loss_type',type=str,help='Loss  used  for  training',default='l1_adv')
    parser.add_argument('--layer', type=str, help='Layer for feature extraction', default='pool1')
    parser.add_argument('--n_songs', type=int, help='Number of songs', default=100)
    parser.add_argument('--override', action='store_true', help='Overwrite existing audio')
    args = parser.parse_args()
    set_name = args.set_name
    model_name = args.model_name
    loss_type = args.loss_type
    layer = args.layer
    n_songs = args.n_songs
    override = args.override

    # Folders
    logmel_predictions_dir = os.path.join(logmel_predictions_root, set_name,model_name + '_' + layer, loss_type)
    emb_dir = os.path.join(emb_root, set_name, model_name + '_' + layer)

    # Output folders
    if not os.path.isdir(logmel_predictions_dir):
        os.makedirs(logmel_predictions_dir)

    # Generate embedding list
    embedding_path_list = glob.glob(os.path.join(emb_dir, '*.npy'))

    # Select number of audio
    if n_songs > 0:
        embedding_path_list = embedding_path_list[0:n_songs]

    # Load tf model
    logmel_predictions_generator = tf.keras.models.load_model(
        os.path.join(saved_models_root,model_name+'_'+layer,loss_type,))

    # Loop over audio
    for embedding_path in tqdm(embedding_path_list):

        # Output path
        logmel_id = os.path.basename(embedding_path).split('.')[0]
        logmel_prediction_path = os.path.join(logmel_predictions_dir, '{:s}.npy'.format(logmel_id))

        if not os.path.isfile(logmel_prediction_path) or override:
            try:
                embedding_chunked = np.load(embedding_path)

                logmel_prediction_chunked = logmel_predictions_generator.predict(embedding_chunked)

                # Generate log-mel spectogram prediction using  selected model

                if len(logmel_prediction_chunked) != 0:
                    np.save(logmel_prediction_path, logmel_prediction_chunked)
            except:
                print('Cannot process file {:s}'.format(logmel_id))



if __name__ == '__main__':
    os.nice(2)
    main()
