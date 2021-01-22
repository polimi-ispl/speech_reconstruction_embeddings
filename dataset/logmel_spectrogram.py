import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from lib import vggish_input
from params import logmel_root

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compute_logmel(audio_path, out_dir, override):

    # Output file path
    audio_id = os.path.basename(audio_path).split('.')[0]
    data_path = os.path.join(out_dir, '{:s}.npy'.format(audio_id))

    # Check if exists
    if not os.path.isfile(data_path) or override:

        try:
            # Compute spectrogram
            logmel_batch = vggish_input.wavfile_to_examples(audio_path)

            # Save
            np.save(data_path, logmel_batch)
        except:
            print('Cannot process file {:s}'.format(audio_id))


def main():
    # Arguments parser
    parser = argparse.ArgumentParser(description='Extract log-mel spectrograms.')
    parser.add_argument('--set_name', type=str, help='Dataset name', default='custom_set')
    parser.add_argument('--audio_dir', type=str, help='Folder containing audio tracks', default='.')
    parser.add_argument('--audio_format', type=str, help='Audio format extension', default='wav')
    parser.add_argument('--n_songs', type=int, help='Number of songs', default=100)
    parser.add_argument('--override', action='store_true', help='Overwrite existing audio')
    args = parser.parse_args()
    set_name = args.set_name
    audio_dir = args.audio_dir
    audio_format = args.audio_format
    n_songs = args.n_songs
    override = args.override

    # Folders
    logmel_dir = os.path.join(logmel_root, set_name)

    # Output folders_

    if not os.path.isdir(logmel_dir):
        os.makedirs(logmel_dir)

    # Generate audio list
    audio_path_list = glob.glob(os.path.join(audio_dir, '*.{:s}'.format(audio_format)))

    # Select number of audio
    if n_songs > 0:
        audio_path_list = audio_path_list[0:n_songs]

    # Loop over audio
    parfun = partial(compute_logmel, out_dir=logmel_dir, override=override)
    pool = Pool(cpu_count() // 2)
    list(tqdm(pool.imap(parfun, audio_path_list), total=len(audio_path_list)))


if __name__ == '__main__':
    os.nice(2)
    main()
