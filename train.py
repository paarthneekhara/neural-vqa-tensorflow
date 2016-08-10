import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse

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
	parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch Size')

	args = parser.parse_args()
	print "Reading QA DATA"
	qa_data = data_loader.load_questions_answers(args)
	
	print "Reading fc7 features"
	fc7_features, image_id_list = data_loader.load_fc7_features(args.data_dir, 'val')
	print "FC7 features", fc7_features.shape
	print "image_id_list", image_id_list.shape

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
	model.build_model()

def get_training_batch(batch_no, batch_size, fc7_features, image_id_list, qa_data, split):
	qa = None
	if split == 'train':
		qa = qa_data['training']
	else:
		qa = qa_data['validation']

	si = batch_no * batch_size
	ei = min(len(qa), si + batch_size)
	n = ei - si
	sentence = np.ndarray( (n, qa_data['max_question_length']), dtype = 'int32')
	answer = np.ndarray( (n, len(qa_data['answer_vocab'])))
	fc7_features = np.ndarray( (n,4096) )
	


if __name__ == '__main__':
	main()