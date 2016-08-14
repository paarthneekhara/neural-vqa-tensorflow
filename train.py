import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_lstm_layers', type=int, default=2,
                       help='num_lstm_layers')
	parser.add_argument('--fc7_feature_length', type=int, default=4096,
                       help='fc7_feature_length')
	parser.add_argument('--rnn_size', type=int, default=512,
                       help='rnn_size')
	parser.add_argument('--embedding_size', type=int, default=512,
                       help='embedding_size'),
	parser.add_argument('--word_emb_dropout', type=float, default=0.5,
                       help='word_emb_dropout')
	parser.add_argument('--image_dropout', type=float, default=0.5,
                       help='image_dropout')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Batch Size')
	parser.add_argument('--epochs', type=int, default=200,
                       help='Expochs')
	parser.add_argument('--debug', type=bool, default=False,
                       help='Debug')
	parser.add_argument('--resume_model', type=str, default=None,
                       help='Trained Model Path')

	args = parser.parse_args()
	print "Reading QA DATA"
	qa_data = data_loader.load_questions_answers(args)
	
	print "Reading fc7 features"
	fc7_features, image_id_list = data_loader.load_fc7_features(args.data_dir, 'train')
	print "FC7 features", fc7_features.shape
	print "image_id_list", image_id_list.shape

	image_id_map = {}
	for i in xrange(len(image_id_list)):
		image_id_map[ image_id_list[i] ] = i

	ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

	model_options = {
		'num_lstm_layers' : args.num_lstm_layers,
		'rnn_size' : args.rnn_size,
		'embedding_size' : args.embedding_size,
		'word_emb_dropout' : args.word_emb_dropout,
		'image_dropout' : args.image_dropout,
		'fc7_feature_length' : args.fc7_feature_length,
		'lstm_steps' : qa_data['max_question_length'] + 1,
		'q_vocab_size' : len(qa_data['question_vocab']),
		'ans_vocab_size' : len(qa_data['answer_vocab'])
	}
	
	
	
	model = vis_lstm_model.Vis_lstm_model(model_options)
	input_tensors, t_loss, t_accuracy, t_p = model.build_model()
	train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(t_loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	
	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)

	for i in xrange(args.epochs):
		batch_no = 0

		while (batch_no*args.batch_size) < len(qa_data['training']):
			sentence, answer, fc7 = get_training_batch(batch_no, args.batch_size, fc7_features, image_id_map, qa_data, 'train')
			_, loss_value, accuracy, pred = sess.run([train_op, t_loss, t_accuracy, t_p], 
				feed_dict={
					input_tensors['fc7']:fc7,
					input_tensors['sentence']:sentence,
					input_tensors['answer']:answer
				}
			)
			batch_no += 1
			if args.debug:
				for idx, p in enumerate(pred):
					print ans_map[p], ans_map[ np.argmax(answer[idx])]

				print "Loss", loss_value, batch_no, i
				print "Accuracy", accuracy
				print "---------------"
			else:
				print "Loss", loss_value, batch_no, i
				print "Training Accuracy", accuracy
			
		save_path = saver.save(sess, "Data/Models/model{}.ckpt".format(i))
		

def get_training_batch(batch_no, batch_size, fc7_features, image_id_map, qa_data, split):
	qa = None
	if split == 'train':
		qa = qa_data['training']
	else:
		qa = qa_data['validation']

	si = (batch_no * batch_size)%len(qa)
	ei = min(len(qa), si + batch_size)
	n = ei - si
	sentence = np.ndarray( (n, qa_data['max_question_length']), dtype = 'int32')
	answer = np.zeros( (n, len(qa_data['answer_vocab'])))
	fc7 = np.ndarray( (n,4096) )

	count = 0
	for i in range(si, ei):
		sentence[count,:] = qa[i]['question'][:]
		answer[count, qa[i]['answer']] = 1.0
		fc7_index = image_id_map[ qa[i]['image_id'] ]
		fc7[count,:] = fc7_features[fc7_index][:]
		count += 1
	
	return sentence, answer, fc7

if __name__ == '__main__':
	main()