import argparse
import glob
import os

import speech_recognition as sr


from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from params import  audio_predictions_root, speech_to_text_predictions_root


# Griffin lim params

# Adjust parameters based on fs


def compute_speech_to_text_prediction(audio_prediction_path,out_dir,override):

  # Output path
  speech_to_text_prediction_id = os.path.basename(audio_prediction_path).split('.')[0]
  speech_to_text_prediction_path = os.path.join(out_dir, '{:s}.txt'.format(speech_to_text_prediction_id))

  if not os.path.isfile(speech_to_text_prediction_path) or override:
    try:
      # Generate speech to text prediction
      r = sr.Recognizer()
      audio_data = sr.AudioFile(audio_prediction_path)
      with audio_data as source:
        audio = r.record(source)
      speech_to_text_prediction = r.recognize_sphinx(audio)


      if len(speech_to_text_prediction) != 0:

        # write to text file
        text_file = open(speech_to_text_prediction_path, "wt")
        n = text_file.write(speech_to_text_prediction)
        text_file.close()
    except:
      print('Cannot process file {:s}'.format(speech_to_text_prediction_id))




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
  speech_to_text_predictions_dir = os.path.join(speech_to_text_predictions_root, set_name, model_name + '_' + layer, loss_type)


  # Output folders
  if not os.path.isdir(speech_to_text_predictions_dir):
    os.makedirs(speech_to_text_predictions_dir)

  # Generate logmel_predictions list
  audio_predictions_path_list = glob.glob(os.path.join(audio_predictions_dir, '*.wav'))

  #logmel_predictions_path_list = sorted(glob.glob(os.path.join(logmel_predictions_dir, '*.npy')))
  #logmel_predictions_path_list = logmel_predictions_path_list[(32 * 400):]

  # Select number of logmel_predictions
  if n_songs > 0:
    audio_predictions_path_list = audio_predictions_path_list[0:n_songs]

  # Loop over logmel predictions
  parfun = partial(compute_speech_to_text_prediction,out_dir=speech_to_text_predictions_dir,override=override)
  pool = Pool(cpu_count() // 4)
  list(tqdm(pool.imap(parfun, audio_predictions_path_list), total=len(audio_predictions_path_list)))


if __name__ == '__main__':
    os.nice(2)
    main()