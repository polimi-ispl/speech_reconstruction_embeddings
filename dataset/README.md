Dataset Preparation
===
Given a set of audio tracks, extract log-mel spectrograms, network output and store everything into TFrectord files.

Folder organization
---
Folders and files are organized as it follows.

    .
    ├── lib                            # Additional python modules
    ├── models                         # TF-Hub models
    ├── embeddings.py                  # Compute embeddings from log-mel spectrograms
    ├── logmel_predictions.py          # Predict log-mel spectrograms from embedding using SavedModels
    ├── audio_predictions.py           # Reconstruct audio from from logmel predictions using griffin-lim
    ├── logmel_spectrogram.py          # Compute log-mel spectrograms
    ├── params.py                      # General parameters
    ├── tfrecrods.py                   # Store embeddings and log-mel spectrograms into TFRecords 
    └── README.md

Usage
---

### Compute log-mel spectrograms
To compute `--n_songs` log-mel spectrograms from dataset `--set_name` composed by `--audio_format` audio files contained in `--audio_dir` run:
```
python logmel_spectrogram.py 
```

Arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| --set_name | `'custom_set'` | Name of the dataset |
| --audio_dir | `'.'` | Directory containing audio files |
| --audio_format | `'wav'` | Audio tracks extension |
| --n_songs | `100` | Howe many tracks to download. Negative to download all songs. |
| --override | `False` | Overwrite existing tracks |

### Compute embeddings
To compute `--n_songs` network features from the `--layer` layer using the dataset `--set_name` and the model `--model_name` stored in `--model_path` run:
```
python embeddings.py 
```

Arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| --set_name | `'custom_set'` | Name of the dataset |
| --model_name | `'MUSAN_small'` | Model name |
| --model_path | `'/nas/home/lcomanducci/audio_feature_reconstruction/models_polimi_small/MUSAN/tf_hub'` | Model path |
| --layer | `'embeddings'` | Output layer name |
| --n_songs | `100` | Howe many tracks to download. Negative to download all songs. |
| --override | `False` | Overwrite existing tracks |

### Save TFRecords
To convert to TFRecord `--n_songs` log-mel spectrograms and embeddings obtained using `--model_name` model to tracks beloning to the `--set_name` dataset run:
```
python tfrecords.py 
```

Arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| --set_name | `'custom_set'` | Name of the dataset |
| --audio_dir | `'.'` | Directory containing audio files |
| --audio_format | `'wav'` | Audio tracks extension |
| --model_name | `'MUSAN_small'` | Model name |
| --layer | `'embeddings'` | Output layer name |
| --n_songs | `100` | Howe many tracks to download. Negative to download all songs. |
| --group | `100` | Number of tracks per TFRecord |
