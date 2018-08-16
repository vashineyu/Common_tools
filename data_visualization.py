# Visualization function
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def img_combine(img, ncols=5, size=1, path=False):
    """
    Draw the images with array
    img: image array to plot - size = n x im_w x im_h x 3
    """
    nimg= img.shape[0]
    nrows=int(ceil(nimg/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols*size,nrows*size))
    if nrows==0:
        return
    elif ncols == 1:
        for r, ax in zip(np.arange(nrows), axes):
            nth=r
            if nth < nimg:
                ax.imshow(img[nth])
            ax.set_axis_off()
    elif nrows==1:
        for c, ax in zip(np.arange(ncols), axes):
            nth=c
            if nth < nimg:
                ax.imshow(img[nth])
            ax.set_axis_off()
    else:
        for r, row in zip(np.arange(nrows), axes):
            for c, ax in zip(np.arange(ncols), row):
                nth=r*ncols+c
                if nth < nimg:
                    ax.imshow(img[nth])
                ax.set_axis_off()
    
    if path:
        plt.tight_layout()
        plt.savefig(path, dpi = 300)
    plt.show()