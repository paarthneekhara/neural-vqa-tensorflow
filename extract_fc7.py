import tensorflow as tf
from scipy import misc
from os import listdir
from os.path import isfile, join
import data_loader
import utils
import argparse
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='val',
                       help='train/val')
	parser.add_argument('--model_path', type=str, default='Data/vgg16.tfmodel',
                       help='Pretrained VGG16 Model')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch Size')

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

	all_data = data_loader.load_questions_answers(args)
	if args.split == "train":
		qa_data = all_data['training']
	else:
		qa_data = all_data['validation']
	
	image_ids = {}
	for qa in qa_data:
		image_ids[qa['image_id']] = 1

	image_id_list = [img_id for img_id in image_ids]
	print "Total Images", len(image_id_list)
	
	
	sess = tf.Session()
	fc7 = np.ndarray( (len(image_id_list), 4096 ) )
	idx = 0
	while idx < len(image_id_list):
		image_batch = np.ndarray( (args.batch_size, 224, 224, 3 ) )
		for i in range(0, args.batch_size):
			image_file = join(args.data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
			image_batch[i,:,:,:] = utils.load_image_array(image_file)
			idx += 1

		feed_dict  = { images : image_batch}
		fc7_tensor = graph.get_tensor_by_name("import/fc7/Reshape:0")
		fc7_batch = sess.run(fc7_tensor, feed_dict = feed_dict)
		fc7[(idx - args.batch_size):idx, :] = fc7_batch
		print "Images Processed", idx
		

	# image_files = [join(mypath, f) for f in os.listdir(args.img_dir) if f.contains(".jpg")]
	


	# for image_file in image_files[0:5]
	# 	img = misc.imread(image_file)
	# 	img_resized = misc.imresize(img, (224, 224))
	# 	print "cat shape", cat_resized.shape
	# 	sess = tf.Session()
	# 	init = tf.initialize_all_variables()
	# 	sess.run( init )

	# 	x = cat_resized.reshape( (1,224, 224, 3) )

	# 	feed_dict  = { images : x}
	# 	prob_tensor = graph.get_tensor_by_name("import/fc7/Reshape:0")
	# 	prob = sess.run( prob_tensor, feed_dict = feed_dict )
	# 	print prob.shape

if __name__ == '__main__':
	main()
