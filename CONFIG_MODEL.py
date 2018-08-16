import tensorflow as tf
from tensorflow import keras as keras
import numpy as np

class PRETRAIN_MODEL_SETUP(object):
    ### model root path
    ### pretrain model of ckpt should be arranged as follow:
    """
    - model_collection_path (folder)
        - resnet_v2_50 (folder)
            - model.ckpt (file)
        - resnet_v2_101
            - model.ckpt
        - OTHER_PRETRAIN_MODEL (folder)
            - model.ckpt
            
    # IF you still want to use pre-train model file in your folder, just direclty write them into dictionary
    """
    model_collection_path = "/data/seanyu/tf-pretrain/"
    
    # PATH TO ckpt file
    PRETRAIN_DICT = {'resnet_50': model_collection_path + '/resnet_v2_50/model.ckpt',
                 'resnet_101': model_collection_path + '/resnet_v2_101/model.ckpt',
                 'resnet_152': model_collection_path + '/resnet_v2_152/model.ckpt',
                 'inception_resnet': model_collection_path + '/inception_resnet_v2/model.ckpt',
                 'densenet_121': model_collection_path + '/densenet_121/model.ckpt',
                 'densenet_169': model_collection_path + '/densenet_169/model.ckpt'
                             }
    
    # image preprocessing function corresponding to each pre-train model
    CORRESPONDING_PREPROC = {
        'resnet_50': tf.keras.applications.resnet50.preprocess_input,
        'resnet_101': tf.keras.applications.resnet50.preprocess_input,
        'resnet_152': tf.keras.applications.resnet50.preprocess_input,
        'inception_resnet': tf.keras.applications.inception_resnet_v2.preprocess_input,
        'densenet_121': tf.keras.applications.densenet.preprocess_input,
        'densenet_169': tf.keras.applications.densenet.preprocess_input
        }
    
    # If you want to do grad-cam, responding layers should be sent
    CORRESPONDING_LAYERS = {
        'resnet_50': 'resnet_v2_50/block4/unit_3/bottleneck_v2/add:0',
        'resnet_101': 'resnet_v2_101/block4/unit_3/bottleneck_v2/add:0',
        'resnet_152': 'resnet_v2_152/block4/unit_3/bottleneck_v2/add:0'
            }