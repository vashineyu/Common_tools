# Visualization function
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from PIL import Image
from scipy.ndimage.filters import gaussian_filter

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


def get_image_for_paper(original_image_object, prediction_map, IHC_map=None,
                        activation_threshold=0.3, overlay_alpha=0.6, sigma_filter=128,
                        mix=False, colormap_style="coolwarm"):
    """
    Get paper used images (raw, overlay_only, raw+overlay, IHC responding region)

    Args:
        - original_image_object: PIL image obejct
        - prediction_map: Array of prediction
        - IHC_map: PIL object of IHC
        - overlap_alpha: control overlay color (0. - 1.0)
        - sigma_filter: Use a Gaussian filter to smooth the prediction map (prevent grid-like looking)
        - mix: True/False, True: return combined map
    Returns:
        Tuple of PIL images
        - (raw, overlay, raw+overlay, IHC)
    """

    # Prediction map filtering
    if sigma_filter > 0:
        pred_smooth = gaussian_filter(prediction_map, sigma=sigma_filter)
    else:
        pred_smooth = prediction_map

    # Create a overlap map
    cm = plt.get_cmap(colormap_style)
    overlay = cm(pred_smooth) * 255
    mr, mc = np.where(pred_smooth > activation_threshold)
    nr, nc = np.where(pred_smooth < activation_threshold)
    overlay[nr, nc, :] = 255
    overlay[nr, nc, 3] = 0
    overlay[mr, mc, 3] = pred_smooth[mr, mc] * 255 * overlay_alpha
    overlay = Image.fromarray(overlay.astype('uint8'))

    # Render overlay to original image
    render = original_image_object.copy()
    render.paste(im=overlay, box=(0, 0), mask=overlay)

    if not mix:
        return (original_image_object, overlay, render, IHC_map)
    else:
        """
        raw         | overlay
        ---------------------
        raw+overlay | IHC
        """
        sz = tuple([int(i / 4) for i in original_image_object.size])
        raw_arr = np.array(original_image_object.resize(sz))  # RGBA
        overlay = np.array(overlay.resize(sz))  # RGBA
        render = np.array(render.resize(sz))  # RGBA
        IHC_map = np.array(IHC_map.resize(sz)) if IHC_map is not None else np.zeros((sz + (4,)))

        r1 = np.hstack((raw_arr, overlay))
        r2 = np.hstack((render, IHC_map))

        mixed = np.vstack((r1, r2))
        return Image.fromarray(mixed.astype('uint8'))
    
def plot_mask_on_image(img, mask, color=[0, 255, 255], alpha=0.3):
    '''Plot colorful masks on the image
    img: cv2 image
    mask: boolean array or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow
    alpha: transparency. float [0, 1]
    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask] = color
    out = cv2.addWeighted(img_layer, alpha, out, 1-alpha, 0, out)
    return out
