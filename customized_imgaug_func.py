import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2

# Customize iaa function
def img_channelswap(images, random_state, parents, hooks):
    for img in images:
        img[:] = do_channel_suffle(img)
    return images

def img_colorswap_func(images, random_state, parents, hooks, ask = None):
    avail_space = {'hsv':cv2.COLOR_RGB2HSV,
                   'hls':cv2.COLOR_RGB2HLS,
                   'lab':cv2.COLOR_RGB2Lab,
                   'luv':cv2.COLOR_RGB2LUV,
                   'xyz':cv2.COLOR_RGB2XYZ,
                   'ycrcb':cv2.COLOR_RGB2YCrCb,
                   'yuv':cv2.COLOR_RGB2YUV}
    for img in images:
        this_swap = avail_space[random.choice(list(avail_space))]
        img[:] = cv2.cvtColor(img, this_swap)
        #plt.imshow(img)
        #plt.show()
    return images

def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


### slave functions

def do_channel_suffle(image):
    img_ = image.copy()
    #r, g, b = img_[:,:,0], img_[:,:,1], img_[:,:,2]
    if img_.shape[-1] == 1:
        raise ValueError("Input should be RGB")
    idx = [0,1,2]
    np.random.shuffle(idx)
    #img_[:,:,idx[0]], img_[:,:,idx[1]], img_[:,:,idx[2]] = r, g, b 
    return img_[:,:,idx]
