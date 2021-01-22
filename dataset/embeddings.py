import argparse
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from params import logmel_root, emb_root, layer_dict

try:
    from models.vggish.vggish import vggish_slim
    from models.vggish.vggish import vggish_params
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    sess = tf.Session(config=config)
except:
    import tensorflow_hub as hub

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    # Arguments parser
    parser = argparse.ArgumentParser(description='Extract embeddings.')
    parser.add_argument('--set_name', type=str, help='Dataset name', default='custom_set')
    parser.add_argument('--model_name', type=str, help='Model name', default='MUSAN_small')
    parser.add_argument('--model_path', type=str, help='Model path', default='/nas/home/lcomanducci/audio_feature_reconstruction/models_polimi_small/MUSAN/tf_hub')
    parser.add_argument('--layer', type=str, help='Layer for feature extraction', default='embeddings')
    parser.add_argument('--n_songs', type=int, help='Number of songs', default=100)
    parser.add_argument('--override', action='store_true', help='Overwrite existing audio')
    args = parser.parse_args()
    set_name = args.set_name
    model_name = args.model_name
    model_path = args.model_path
    layer = args.layer
    n_songs = args.n_songs
    override = args.override


    # Folders
    logmel_dir = os.path.join(logmel_root, set_name)
    emb_dir = os.path.join(emb_root, set_name, model_name + '_' + layer)

    # Output folders
    if not os.path.isdir(emb_dir):
        os.makedirs(emb_dir)

    # Generate logmel list
    logmel_path_list = glob.glob(os.path.join(logmel_dir, '*.npy'))

    # Select number of audio
    if n_songs > 0:
        logmel_path_list = logmel_path_list[0:n_songs]

    # Load tf model
    if model_name == 'VGGish':
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, os.path.join(model_path, 'vggish_model.ckpt'))
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(layer_dict[model_name][layer])
    else:
        embedding_generator = hub.load(model_path, tags='batch_norm_fixed')

    # Loop over audio
    for logmel_path in tqdm(logmel_path_list):

        # Output path
        logmel_id = os.path.basename(logmel_path).split('.')[0]
        emb_path = os.path.join(emb_dir, '{:s}.npy'.format(logmel_id))

        if not os.path.isfile(emb_path) or override:

            try:
                logmel_chunked = np.load(logmel_path)
                if model_name == 'VGGish':
                    embedding_chunked = sess.run([embedding_tensor], feed_dict={features_tensor: logmel_chunked})
                    if len(np.asarray(embedding_chunked).shape) > 2:
                        embedding_chunked = np.asarray(embedding_chunked)[0]
                    else:
                        embedding_chunked = np.asarray(embedding_chunked)
                else:

                    logmel_chunked_tf = tf.convert_to_tensor(logmel_chunked, dtype=tf.float32)
                    embedding_chunked = embedding_generator.signatures[layer_dict[model_name][layer]](logmel_chunked_tf)
                    embedding_chunked = np.asarray(embedding_chunked['default'])

                if len(embedding_chunked) != 0:
                    np.save(emb_path, embedding_chunked)
            except:
                print('Cannot process file {:s}'.format(logmel_id))
                print(logmel_path)


if __name__ == '__main__':
    os.nice(2)
    main()
