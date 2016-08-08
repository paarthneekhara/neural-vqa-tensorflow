import tensorflow as tf
from scipy import misc
from os import listdir
from os.path import isfile, join
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='val',
                       help='train/val')
	parser.add_argument('--model_path', type=str, default='Data/vgg16.tfmodel',
                       help='Pretrained VGG16 Model')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')

	args = parser.parse_args()
	
	vgg_file = open(args.model_path)
	vgg16raw = vgg_file.read()
	vgg_file.close()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)

	images = tf.placeholder("float", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={ "images": images })

	graph = tf.get_default_graph()
	for opn in graph.get_operations():
		print "Name", opn.name, opn.values()

	
	image_files = [join(mypath, f) for f in os.listdir(args.img_dir) if f.contains(".jpg")]
	


	for image_file in image_files[0:5]
		cat = misc.imread(image_file)
		cat_resized = misc.imresize(cat, (224, 224))
		print "cat shape", cat_resized.shape
		sess = tf.Session()
		init = tf.initialize_all_variables()
		sess.run( init )

		x = cat_resized.reshape( (1,224, 224, 3) )

		feed_dict  = { images : x}
		prob_tensor = graph.get_tensor_by_name("import/fc7/Reshape:0")
		prob = sess.run( prob_tensor, feed_dict = feed_dict )
		print prob.shape

if __name__ == '__main__'