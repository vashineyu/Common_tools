# Common_tools
Python tools that I commonly used in most of projects. There tools include visualization, quick performance summaization, corresponding tensorflow-transfer model preprocessing functions, and other handy tools.

# Installation
Just clone and sys append it will be fine.
```
git clone https://github.com/vashineyu/Common_tools
```
And assume you clone to your home folder
```
import sys
sys.path.append("~/Common_tools")
```

# Table of Contents
The following contents document the main purpose of each py the highlight/personal common used function.
<To be done>

main function|purpose|description|highlight/most used function
-------------|------:|----------:|----------------------------:
data_visualization | Visualization| Useful when check stacked images | img_combine: plot a stack of images into grid-like
Recording | Documentation/Recording | Used to record experiment setting | Experiment_Recoding is a class for you to put experiment related configs/setting into a new folder.  
result_summary | Visualization | Compute metric after your experiment done | pdml is a good package!
customized_imgaug_func | data operation | Custom augmentation | includes special image augmentations such as channel shift & color space swap
CONFIG_MODEL | preprocessing | record pretrain models and corresponding preprocessing functions / visualization layers | Remember to check and change the model_collection_path inside the class
keras_callbacks | model callbacks | callback function when training | AUC metric logger
tf_callbacks | model callbacks | implementation of most used callbacks in Keras. | tensorflow earlystop / model checkpoint / reduce learning rate 
model_visualization | Visualization | implement Gradient-Class-Activation-Map with tensorflow | ---
slice_reader | data operation | digital pathology data format handler | actually it should be named "slide_reader..."
misc | misc | misc | something I don't know where to put it


