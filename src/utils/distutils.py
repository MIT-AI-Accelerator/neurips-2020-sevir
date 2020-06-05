import tensorflow as tf
import tensorflow.keras.backend as K

def setup_gpu_mapping(hvd):
    # following is for TF 1.14
    if tf.version.VERSION == '1.14.0':
        print('=> setting gpu options for TF 1.14')
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))
    else:
        print('setting gpu options for TF 2.1')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    return

