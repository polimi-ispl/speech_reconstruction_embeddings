import argparse
import glob
import os

import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm
from lib import vggish_params

from functools import partial
from multiprocessing import Pool, cpu_count
from params import logmel_predictions_root, audio_predictions_root


# Griffin lim params

# Adjust parameters based on fs
window_length_samples = int(round(vggish_params.SAMPLE_RATE * vggish_params.STFT_WINDOW_LENGTH_SECONDS))
hop_length_samples = int(round(vggish_params.SAMPLE_RATE  * vggish_params.STFT_HOP_LENGTH_SECONDS))
fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))


def griffin_lim_VGGish(estimated_spectrogram):
  """
  Reconstructs audio signal from log-mel spectrogram using griffin-lim
  Args
    estimated_spectrogram: Float  array or list, log-mel spectrogram
  Returns
    t_hat_vg: Float  array or list, time axis of reconstructed audio signal
    x_hat_vg: Float  array or list, reconstructed audio signal
  """
  griffin_lim_input = estimated_spectrogram  #
  x_hat_vg = librosa.feature.inverse.mel_to_audio(
    np.exp(griffin_lim_input) - vggish_params.LOG_OFFSET,
    hop_length=hop_length_samples,
    win_length=window_length_samples,
    window='hann',
    n_fft=fft_length,
    fmin=vggish_params.MEL_MIN_HZ,
    fmax=vggish_params.MEL_MAX_HZ,
    htk=True,
    norm=None,
    power=1,
    sr=vggish_params.SAMPLE_RATE
  )
  t_hat_vg = np.arange(len(x_hat_vg)) / vggish_params.SAMPLE_RATE
  return t_hat_vg, x_hat_vg


def compute_audio_prediction(logmel_prediction_path,out_dir,override):

  # Output path
  audio_prediction_id = os.path.basename(logmel_prediction_path).split('.')[0]
  audio_prediction_path = os.path.join(out_dir, '{:s}.wav'.format(audio_prediction_id))

  if not os.path.isfile(audio_prediction_path) or override:
    try:
      logmel_prediction_chunked = np.load(logmel_prediction_path)
      print(logmel_prediction_path)

      # Generate log-mel spectogram prediction using  selected model

      # Preprocess
      logmel_prediction_chunked = np.reshape(
        logmel_prediction_chunked,
        (logmel_prediction_chunked.shape[0]*logmel_prediction_chunked.shape[1],logmel_prediction_chunked.shape[2]))*10

      # VGGish
      _, audio_prediction = griffin_lim_VGGish(logmel_prediction_chunked.transpose())

      if len(audio_prediction) != 0:

        #wavwrite(audio_prediction_path,vggish_params.SAMPLE_RATE,audio_prediction)
        sf.write(
          audio_prediction_path, audio_prediction, vggish_params.SAMPLE_RATE, 'PCM_16')

    except:
      print('Cannot process file {:s}'.format(audio_prediction_id))


def main():
  # Arguments parser
  parser = argparse.ArgumentParser(description='Extract Log-mel spectrograms predictions.')
  parser.add_argument('--set_name', type=str, help='Dataset name', default='lj_speech')
  parser.add_argument('--model_name', type=str, help='Model name', default='VGGish')
  parser.add_argument('--loss_type', type=str, help='Loss  used  for  training', default='l1_adv')
  parser.add_argument('--layer', type=str, help='Layer for feature extraction', default='pool1')
  parser.add_argument('--n_songs', type=int, help='Number of songs', default=1)
  parser.add_argument('--override', action='store_true', help='Overwrite existing audio')
  args = parser.parse_args()
  set_name = args.set_name
  model_name = args.model_name
  loss_type = args.loss_type
  layer = args.layer
  n_songs = args.n_songs
  override = args.override



  # Folders
  audio_predictions_dir = os.path.join(audio_predictions_root, set_name, model_name + '_' + layer, loss_type)
  logmel_predictions_dir = os.path.join(logmel_predictions_root, set_name, model_name + '_' + layer, loss_type)

  # Output folders
  if not os.path.isdir(audio_predictions_dir):
    os.makedirs(audio_predictions_dir)

  # Generate logmel_predictions list
  logmel_predictions_path_list = glob.glob(os.path.join(logmel_predictions_dir, '*.npy'))

  logmel_predictions_path_list = sorted(glob.glob(os.path.join(logmel_predictions_dir, '*.npy')))
  logmel_predictions_path_list = logmel_predictions_path_list[(32 * 400):]

  # Select number of logmel_predictions
  if n_songs > 0:
    logmel_predictions_path_list = logmel_predictions_path_list[0:n_songs]

  # Loop over logmel predictions
  parfun = partial(compute_audio_prediction,out_dir=audio_predictions_dir,override=override)
  pool = Pool(cpu_count() // 2)
  list(tqdm(pool.imap(parfun, logmel_predictions_path_list), total=len(logmel_predictions_path_list)))


if __name__ == '__main__':
    os.nice(2)
    main()
