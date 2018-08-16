import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

def grad_cam_tf(model, im, cls_select, tf_sess, layer, alpha = 0.6, preproc_function = None, reverse_function = None):
    image = im.copy()
    if len(image) != 4:
        # Make image batch-like
        image = image[np.newaxis, :,:,:]
    
    if reverse_function is not None:
        image_original = reverse_function(image[0])
    else:
        image_original = image[0]
    
    if preproc_function is not None:
        # this preprocessing function generally apply to 4-D array
        # TO DO: make it tolerate to 3-D input
        image = preproc_function(image.astype('float32'))
        
    H, W = image.shape[1], image.shape[2]
    
    y_c = model.model_ops['output']['prediction1'][0, cls_select]
    conv_output = model.sess.graph.get_tensor_by_name(layer)
    grads = tf.gradients(y_c, conv_output)
    
    output, grads_val = tf_sess.run([conv_output, grads],
                                    feed_dict = {model.model_ops['input'][0]: image,
                                                 model.model_ops['is_training']: False,
                                                 tf.keras.backend.learning_phase(): False})
    #return output, grads_val
    output, grads_val = output[0, :], grads_val[0][0, :,:,:]
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.dot(output, weights) 

    ## resize it ##
    cam = tf.image.resize_bicubic(images=cam[np.newaxis, : , :, np.newaxis], 
                                  size=(H, W))
    cam = cam[0, :, :, 0]
    cam = tf.maximum(cam, 0)
    cam = cam / tf.reduce_max(cam)
    cam = tf_sess.run(cam)
    
    # apply color map
    mapping = cv2.applyColorMap(np.uint8(255 * (1-cam) ), cv2.COLORMAP_JET)
    #mapping = cv2.GaussianBlur(mapping, (3,3), 1)
    mapping = np.concatenate((mapping, ((mapping.max(axis=-1) - 128 )*255*alpha)[:,:,np.newaxis]), axis = -1)
    
    # foreground - background
    background = Image.fromarray(image_original)
    foreground = Image.fromarray(mapping.astype('uint8'))
    background.paste(foreground, (0, 0), foreground)
    
    return cam, background