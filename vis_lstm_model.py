import tensorflow as tf
import math

class Vis_lstm_model:
	def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
		return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

	def init_bias(self, dim_out, name=None):
		return tf.Variable(tf.zeros([dim_out]), name=name)

	def __init__(self, options):
		with tf.device('/cpu:0'):
			self.options = options

			# +1 for zero padding
			self.Wemb = tf.Variable(tf.random_uniform([options['q_vocab_size'] + 1, options['embedding_size']], -1.0, 1.0), name = 'Wemb')
			self.Wimg = self.init_weight(options['fc7_feature_length'], options['embedding_size'], name = 'Wimg')
			self.bimg = self.init_bias(options['embedding_size'], name = 'bimg')
			
			# TODO: Assumed embedding size and rnn-size to be same
			self.lstm_W = []
			self.lstm_U = []
			self.lstm_b = []
			for i in range(options['num_lstm_layers']):
				W = self.init_weight(options['rnn_size'], 4 * options['rnn_size'], name = ('rnnw_' + str(i)))
				U = self.init_weight(options['rnn_size'], 4 * options['rnn_size'], name = ('rnnu_' + str(i)))
				b = self.init_bias(4 * options['rnn_size'], name = ('rnnb_' + str(i)))
				self.lstm_W.append(W)
				self.lstm_U.append(U)
				self.lstm_b.append(b)

			self.ans_sm_W = self.init_weight(options['rnn_size'], options['ans_vocab_size'], name = 'ans_sm_W')
			self.ans_sm_b = self.init_bias(options['ans_vocab_size'], name = 'ans_sm_b')

	def forward_pass_lstm(self, word_embeddings):
		x = word_embeddings
		output = None
		for l in range(self.options['num_lstm_layers']):
			h = [None for i in range(self.options['lstm_steps'])]
			c = [None for i in range(self.options['lstm_steps'])]
			layer_output = []
			for lstm_step in range(self.options['lstm_steps']):
				if lstm_step == 0:
					lstm_preactive = tf.matmul(x[lstm_step], self.lstm_W[l]) + self.lstm_b[l]
				else:
					lstm_preactive = tf.matmul(h[lstm_step-1], self.lstm_U[l]) + tf.matmul(x[lstm_step], self.lstm_W[l]) + self.lstm_b[l]
				i, f, o, new_c = tf.split(1, 4, lstm_preactive)
				i = tf.nn.sigmoid(i)
				f = tf.nn.sigmoid(f)
				o = tf.nn.sigmoid(o)
				new_c = tf.nn.tanh(new_c)
				
				if lstm_step == 0:
					c[lstm_step] = i * new_c
				else:
					c[lstm_step] = f * c[lstm_step-1] + i * new_c

				# BUG IN THE LSTM --> Haven't corrected this yet, Will have to retrain the model.
				# h[lstm_step] = o * tf.nn.tanh(c[lstm_step])
				h[lstm_step] = o * tf.nn.tanh(new_c)
				layer_output.append(h[lstm_step])

			x = layer_output
			output = layer_output

		return output




	def build_model(self):
		fc7_features = tf.placeholder('float32',[ None, self.options['fc7_feature_length'] ], name = 'fc7')
		sentence = tf.placeholder('int32',[None, self.options['lstm_steps'] - 1], name = "sentence")
		answer = tf.placeholder('float32', [None, self.options['ans_vocab_size']], name = "answer")


		word_embeddings = []
		for i in range(self.options['lstm_steps']-1):
			word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])
			word_emb = tf.nn.dropout(word_emb, self.options['word_emb_dropout'], name = "word_emb" + str(i))
			word_embeddings.append(word_emb)

		image_embedding = tf.matmul(fc7_features, self.Wimg) + self.bimg
		image_embedding = tf.nn.tanh(image_embedding)
		image_embedding = tf.nn.dropout(image_embedding, self.options['image_dropout'], name = "vis_features")

		# Image as the last word in the lstm
		word_embeddings.append(image_embedding)
		lstm_output = self.forward_pass_lstm(word_embeddings)
		lstm_answer = lstm_output[-1]
		logits = tf.matmul(lstm_answer, self.ans_sm_W) + self.ans_sm_b
		ce = tf.nn.softmax_cross_entropy_with_logits(logits, answer, name = 'ce')
		answer_probab = tf.nn.softmax(logits, name='answer_probab')
		
		predictions = tf.argmax(answer_probab,1)
		correct_predictions = tf.equal(tf.argmax(answer_probab,1), tf.argmax(answer,1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

		loss = tf.reduce_sum(ce, name = 'loss')
		input_tensors = {
			'fc7' : fc7_features,
			'sentence' : sentence,
			'answer' : answer
		}
		return input_tensors, loss, accuracy, predictions

	def build_generator(self):
		fc7_features = tf.placeholder('float32',[ None, self.options['fc7_feature_length'] ], name = 'fc7')
		sentence = tf.placeholder('int32',[None, self.options['lstm_steps'] - 1], name = "sentence")

		word_embeddings = []
		for i in range(self.options['lstm_steps']-1):
			word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])
			word_embeddings.append(word_emb)

		image_embedding = tf.matmul(fc7_features, self.Wimg) + self.bimg
		image_embedding = tf.nn.tanh(image_embedding)

		word_embeddings.append(image_embedding)
		lstm_output = self.forward_pass_lstm(word_embeddings)
		lstm_answer = lstm_output[-1]
		logits = tf.matmul(lstm_answer, self.ans_sm_W) + self.ans_sm_b
		
		answer_probab = tf.nn.softmax(logits, name='answer_probab')
		
		predictions = tf.argmax(answer_probab,1)

		input_tensors = {
			'fc7' : fc7_features,
			'sentence' : sentence
		}

		return input_tensors, predictions, answer_probab


