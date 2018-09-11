import numpy as np
import cv2
from openslide import open_slide, ImageSlide
from PIL import Image


class SLIDE_OPENER():
    def __init__(self, this_slide, cv_filter_module = 'cv_contour', show_info = True, do_init = True):
        """
        Initalize the slider viewer
        top level: largest level
        bottom level: detailed level
        """
        if show_info:
            print("Reading whole slide image: %s" % this_slide)
        self.current_slide = open_slide(this_slide)
        self.filter_method = cv_filter_module
        self.show_info = show_info
        self.do_init = do_init
        self.init_processing()
        open_level = 8
        n_full_patches = self.levels_dimension[open_level-1][0] * self.levels_dimension[open_level - 1][1] # we take the 8th level (due to process)
        if show_info:
            print("Read done, #total candidate patches: %i out of total patches: %i" % (self.n_total_patch, n_full_patches))
        
    def init_processing(self):
        """
        Initalizer step
        - Get positive mask of the top level
        - Get numbers of levels
        """
        self.numbers_of_levels = self.current_slide.level_count
        self.patch_size_of_levels = [2**i for i in range(self.numbers_of_levels)]
        self.levels_dimension = self.current_slide.level_dimensions
        if self.do_init:
            self.positive_array, coords = self._get_contour(top_level = 8)
            self.h_coords, self.w_coords = coords
            self.n_total_patch = len(self.h_coords)
    
    def _get_contour(self, top_level=8):
        """
        bottom level = 8 will make zoom in resolution as 128 x 128 (2^(n-1))
        """
        tmp = self.current_slide.read_region(level= top_level-1, #self.numbers_of_levels-1, 
                                             location=(0,0), 
                                             size = self.levels_dimension[top_level-1] # self.levels_dimension[-1]
                                            )
        tmp = np.array(tmp)
        self.im_gray = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
        if self.filter_method == 'cv_contour':
            # Use filtering module
            

            ret, thresh = cv2.threshold(self.im_gray, 127, 240, 1)
            im_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            arr = cv2.drawContours(im_, contours, -1, color = (255), thickness = 4)
            coordinate_bound = np.where(arr != 0)
        else:
            # Skip filtering module
            if self.show_info:
                print("Skip filtering, use all patches")
            arr = np.ones(self.im_gray.shape)
            coordinate_bound = np.where(arr != 0)
        
        return arr, coordinate_bound
    
    def get_coordinates(self):
        """
        Get coordinates handler (on the top level)
        
        :Return
            - coord_w, coord_h
        """
        return self.w_coords, self.h_coords
    
    def get_single_zoomed_patch(self, patch_id_or_coord, size = (128, 128), levels = 0, top_level = 8):
        """
        Get patch image
        Generally, patch size should be 2^k (k=levels)
        
        :Input
            - patch_id_or_coord: if a single integer, take coords from w_coord/h_coord. else: tuple of coordinate(w/h) of the top level
            - size: patch size (w, h)
            - levels: 0 as bottom level
            - viewer: if True, plt show image. (DONT USE in inference mode)
        
        :Return
            - single patch image (dont throw out whole patch images, to optimize inference speed with generator)
            - coordinate on top level
        """
        if type(patch_id_or_coord) is int:
            w = self.w_coords[patch_id_or_coord]
            h = self.h_coords[patch_id_or_coord]
        elif type(patch_id_or_coord) is tuple:
            w, h = patch_id_or_coord
        else:
            raise AssertionError('patch_id_or_coord should be either integer that indicate patch_id or tuple of w/h coordinate')
        
        patch_zoom_in_multiply = int(self.levels_dimension[levels][0] / self.levels_dimension[top_level-1][0])
        
        this_patch = self.current_slide.read_region(level=levels, 
                                                   location=(w*patch_zoom_in_multiply,h*patch_zoom_in_multiply), 
                                                   size = (patch_zoom_in_multiply, patch_zoom_in_multiply))
        
        return np.array(this_patch), (w, h)
    
    def show_snapshot(self):
        # show snapshot of top level
        try:
            import matplotlib.pyplot as plt
            from ipywidgets import IntSlider, interactive
        except:
            print("Only available with ipynb")
            
        original_img = np.array(self.current_slide.read_region(level=self.numbers_of_levels-1, 
                                                          location=(0,0), 
                                                          size = self.levels_dimension[-1]))
        plt.figure(figsize = (8,8))
        plt.imshow(original_img)
        plt.axis('off')
        plt.show()
    
    def viewer(self, index = 0, levels = 0, rect_hl = 6, border_size = 2, bottom_level = -1, prediction_result = None):
        """
        Open an interactive viewer
        Only support indexing & jupyter notebook viewing
        
        - prediction result: an prediction array (n batch x n categories)
        """
        try:
            import matplotlib.pyplot as plt
            from ipywidgets import IntSlider, interactive
        except:
            print("Only available with ipynb")
        # Whole slide reference
        original_img = np.array(self.current_slide.read_region(level=self.numbers_of_levels-1, 
                                                          location=(0,0), 
                                                          size = self.levels_dimension[-1]))
        inner_hl = rect_hl - border_size
        
        def display_image(index):
            
            if prediction_result is not None:
                this_predict = prediction_result[index]
            
            ori_img = original_img.copy()
            ref_img = self.positive_array.copy()
            
            # Patch
            w = self.w_coords[index]
            h = self.h_coords[index]
            patch_zoom_in_multiply = int(self.levels_dimension[levels][0] / self.levels_dimension[bottom_level][0])

            img = self.current_slide.read_region(level=levels, 
                                                 location=(w*patch_zoom_in_multiply,h*patch_zoom_in_multiply), 
                                                 size = (patch_zoom_in_multiply, patch_zoom_in_multiply))
            
            saving_region = ori_img[(h-inner_hl):(h+inner_hl),
                                    (w-inner_hl):(w+inner_hl)].copy()
            
            ori_img[(h-rect_hl):(h+rect_hl),(w-rect_hl):(w+rect_hl), ...] = [0,0,255,255]
            ori_img[(h-inner_hl):(h+inner_hl),(w-inner_hl):(w+inner_hl), ...] = saving_region
            
            plt.figure(figsize=(8,8))
            plt.imshow(ori_img)
            plt.axis('off')
            plt.show()
            
            plt.figure(figsize=(8,8))
            plt.imshow(ref_img)
            plt.axis('off')
            plt.show()
            
            plt.imshow(np.array(img))
            if prediction_result is not None:
                msg = '|'.join(['%.5f' % (i) for i in this_predict])
                plt.title(msg)
            plt.axis('off')
            plt.show()
        
        return interactive(display_image, index=IntSlider(min=0,max=self.n_total_patch-1,step=1,value=0))