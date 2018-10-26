from collections import deque
import numpy as np
import random

class Memory_buffer():
    """
    This class is used to store bad example with a queue. so that we can force the model train on them again 
    """
    def __init__(self, queue_size = 1000):
        """
        queue_size: how many images to keep
        """
        self.image_queue = deque(maxlen = queue_size)
        self.label_queue = deque(maxlen = queue_size)
        
    def put_items(self, image_array, label_array, index_array):
        """
        image_array: should be a 4D array (batch, w, h, c)
        label_array: should be a 2D array (batch, one-hot)
        index_array: should be a 1D array of indexes that identify which image should be put into queue
        """
        random.shuffle(index_array)
        for i in index_array:
            self.image_queue.append(image_array[i])
            self.label_queue.append(label_array[i])
    
    def get_items(self, n = 12):
        """
        Get item from queue, follow "first in, first out" rule
        """
        x_ = np.array([self.image_queue.popleft() for _ in range(n)])
        y_ = np.array([self.label_queue.popleft() for _ in range(n)])
        
        return x_, y_
    
    def current_size(self):
        return len(self.image_queue)
