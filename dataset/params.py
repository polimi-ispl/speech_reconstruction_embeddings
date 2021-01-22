logmel_root = '/nas/public/exchange/audio_feature_reconstruction/logmel'
emb_root = '/nas/public/exchange/audio_feature_reconstruction/embeddings'
tf_root = '/nas/public/exchange/audio_feature_reconstruction/tfrecords'
logmel_predictions_root = '/nas/public/exchange/audio_feature_reconstruction/logmel_predictions'
saved_models_root = '/nas/public/exchange/audio_feature_reconstruction/models'
audio_predictions_root = '/nas/public/exchange/audio_feature_reconstruction/audio_predictions'
speech_to_text_predictions_root = '/nas/public/exchange/audio_feature_reconstruction/speech_to_text_predictions'

layer_dict = dict()
layer_dict['VGGish'] = dict()
layer_dict['VGGish']['pool1'] = 'vggish/pool1/MaxPool:0'
layer_dict['VGGish']['pool2'] = 'vggish/pool2/MaxPool:0'
layer_dict['VGGish']['pool3'] = 'vggish/pool3/MaxPool:0'
layer_dict['VGGish']['pool4'] = 'vggish/pool4/MaxPool:0'
layer_dict['VGGish']['fc1_1'] = 'vggish/fc1/fc1_1/Relu:0'
layer_dict['VGGish']['fc1_2'] = 'vggish/fc1/fc1_2/Relu:0'
layer_dict['VGGish']['fc2'] = 'vggish/fc2/Relu:0'
layer_dict['VGGish']['embeddings'] = 'vggish/embedding:0'

layer_dict['MUSAN'] = dict()
layer_dict['MUSAN']['layer_0'] = 'layer_0'
layer_dict['MUSAN']['layer_1'] = 'layer_1'
layer_dict['MUSAN']['layer_2'] = 'layer_2'
layer_dict['MUSAN']['layer_3'] = 'layer_3'
layer_dict['MUSAN']['layer_4'] = 'layer_4'
layer_dict['MUSAN']['layer_5'] = 'layer_5'
layer_dict['MUSAN']['layer_6'] = 'layer_6'
layer_dict['MUSAN']['layer_7'] = 'layer_7'

layer_dict['TUT-urban-acoustic-scenes-2018'] = dict()
layer_dict['TUT-urban-acoustic-scenes-2018']['layer_0'] = 'layer_0'
layer_dict['TUT-urban-acoustic-scenes-2018']['layer_1'] = 'layer_1'
layer_dict['TUT-urban-acoustic-scenes-2018']['layer_2'] = 'layer_2'
layer_dict['TUT-urban-acoustic-scenes-2018']['layer_3'] = 'layer_3'
layer_dict['TUT-urban-acoustic-scenes-2018']['layer_4'] = 'layer_4'
layer_dict['TUT-urban-acoustic-scenes-2018']['layer_5'] = 'layer_5'
layer_dict['TUT-urban-acoustic-scenes-2018']['layer_6'] = 'layer_6'
layer_dict['TUT-urban-acoustic-scenes-2018']['layer_7'] = 'layer_7'

layer_dict['birdsong_detection'] = dict()
layer_dict['birdsong_detection']['layer_0'] = 'layer_0'
layer_dict['birdsong_detection']['layer_1'] = 'layer_1'
layer_dict['birdsong_detection']['layer_2'] = 'layer_2'
layer_dict['birdsong_detection']['layer_3'] = 'layer_3'
layer_dict['birdsong_detection']['layer_4'] = 'layer_4'
layer_dict['birdsong_detection']['layer_5'] = 'layer_5'
layer_dict['birdsong_detection']['layer_6'] = 'layer_6'
layer_dict['birdsong_detection']['layer_7'] = 'layer_7'

layer_dict['lang_id'] = dict()
layer_dict['lang_id']['layer_0'] = 'layer_0'
layer_dict['lang_id']['layer_1'] = 'layer_1'
layer_dict['lang_id']['layer_2'] = 'layer_2'
layer_dict['lang_id']['layer_3'] = 'layer_3'
layer_dict['lang_id']['layer_4'] = 'layer_4'
layer_dict['lang_id']['layer_5'] = 'layer_5'
layer_dict['lang_id']['layer_6'] = 'layer_6'
layer_dict['lang_id']['layer_7'] = 'layer_7'

layer_dict['speech_commands_v1'] = dict()
layer_dict['speech_commands_v1']['layer_0'] = 'layer_0'
layer_dict['speech_commands_v1']['layer_1'] = 'layer_1'
layer_dict['speech_commands_v1']['layer_2'] = 'layer_2'
layer_dict['speech_commands_v1']['layer_3'] = 'layer_3'
layer_dict['speech_commands_v1']['layer_4'] = 'layer_4'
layer_dict['speech_commands_v1']['layer_5'] = 'layer_5'
layer_dict['speech_commands_v1']['layer_6'] = 'layer_6'
layer_dict['speech_commands_v1']['layer_7'] = 'layer_7'

layer_dict['speech_commands_v2'] = dict()
layer_dict['speech_commands_v2']['layer_0'] = 'layer_0'
layer_dict['speech_commands_v2']['layer_1'] = 'layer_1'
layer_dict['speech_commands_v2']['layer_2'] = 'layer_2'
layer_dict['speech_commands_v2']['layer_3'] = 'layer_3'
layer_dict['speech_commands_v2']['layer_4'] = 'layer_4'
layer_dict['speech_commands_v2']['layer_5'] = 'layer_5'
layer_dict['speech_commands_v2']['layer_6'] = 'layer_6'
layer_dict['speech_commands_v2']['layer_7'] = 'layer_7'


layer_dict['multi_7'] = dict()
layer_dict['multi_7']['layer_0'] = 'layer_0'
layer_dict['multi_7']['layer_1'] = 'layer_1'
layer_dict['multi_7']['layer_2'] = 'layer_2'
layer_dict['multi_7']['layer_3'] = 'layer_3'
layer_dict['multi_7']['layer_4'] = 'layer_4'
layer_dict['multi_7']['layer_5'] = 'layer_5'
layer_dict['multi_7']['layer_6'] = 'layer_6'
layer_dict['multi_7']['layer_7'] = 'layer_7'


layer_dict['librispeech'] = dict()
layer_dict['librispeech']['layer_0'] = 'layer_0'
layer_dict['librispeech']['layer_1'] = 'layer_1'
layer_dict['librispeech']['layer_2'] = 'layer_2'
layer_dict['librispeech']['layer_3'] = 'layer_3'
layer_dict['librispeech']['layer_4'] = 'layer_4'
layer_dict['librispeech']['layer_5'] = 'layer_5'
layer_dict['librispeech']['layer_6'] = 'layer_6'
layer_dict['librispeech']['layer_7'] = 'layer_7'