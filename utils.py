import numpy as np
from scipy import misc

# VGG 16 accepts RGB channel 0 to 1 (This tensorflow model).
def load_image_array(image_file):
	img = misc.imread(image_file)
	img_resized = misc.imresize(img, (224, 224))
	return (img_resized/255.0).astype('float32')