import numpy as np
from scipy import misc
import tensorflow as tf

# VGG 16 accepts RGB channel 0 to 1 (This tensorflow model).
def load_image_array(image_file):
	img = misc.imread(image_file)
	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new

	img_resized = misc.imresize(img, (224, 224))
	return (img_resized/255.0).astype('float32')

# FOR PREDICTION ON A SINGLE IMAGE
def extract_fc7_features(image_path, model_path):
	vgg_file = open(model_path)
	vgg16raw = vgg_file.read()
	vgg_file.close()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)
	images = tf.placeholder("float32", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={ "images": images })
	graph = tf.get_default_graph()

	sess = tf.Session()
	image_array = load_image_array(image_path)
	image_feed = np.ndarray((1,224,224,3))
	image_feed[0:,:,:] = image_array
	feed_dict  = { images : image_feed }
	fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
	fc7_features = sess.run(fc7_tensor, feed_dict = feed_dict)
	sess.close()
	return fc7_features