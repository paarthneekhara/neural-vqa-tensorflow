import tensorflow as tf
import math

class Vis_lstm_model:
	def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
		return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

	def init_bias(self, dim_out, name=None):
		return tf.Variable(tf.zeros([dim_out]), name=name)

	def __init__(self, options):
		self.options = options
		self.Wemb = tf.Variable(tf.random_uniform([options['q_vocab_size'], options['embedding_size']], -1.0, 1.0), name = 'Wemb')
		self.Wimg = self.init_weight(options['fc7_feature_length'], options['embedding_size'], name = 'Wimg')
		self.bimg = self.init_bias(options['embedding_size'], name = 'bimg')

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
			h = 0
			c = 0
			layer_output = []
			for lstm_step in range(self.options['lstm_steps']):
				if lstm_step == 0:
					lstm_preactive = tf.matmul(x[lstm_step], self.lstm_W[l]) + self.lstm_b[l]
				else:
					lstm_preactive = tf.matmul(h, self.lstm_U[l]) + tf.matmul(x[lstm_step], self.lstm_W[l]) + self.lstm_b[l]
				i, f, o, new_c = tf.split(1, 4, lstm_preactive)
				i = tf.nn.sigmoid(i)
				f = tf.nn.sigmoid(f)
				o = tf.nn.sigmoid(o)
				new_c = tf.nn.tanh(new_c)
				c = f * c + i * new_c
				h = o * tf.nn.tanh(new_c)
				layer_output.append(h)
			x = layer_output
			output = layer_output

		return output




	def build_model(self):
		fc7_features = tf.placeholder('float32',[ None, self.options['fc7_feature_length'] ])
		sentence = tf.placeholder('int32',[None, self.options['lstm_steps']] )
		answer = tf.placeholder('float32', [None, self.options['ans_vocab_size']])


		word_embeddings = []
		for i in range(self.options['lstm_steps']-1):
			word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])
			word_emb = tf.nn.dropout(word_emb, self.options['word_emb_dropout'])
			word_embeddings.append(word_emb)

		image_embedding = tf.matmul(fc7_features, self.Wimg) + self.bimg
		image_embedding = tf.nn.tanh(image_embedding)
		image_embedding = tf.nn.dropout(image_embedding, self.options['image_dropout'])

		# Image as the last word in the lstm
		word_embeddings.append(image_embedding)
		lstm_output = self.forward_pass_lstm(word_embeddings)
		lstm_answer = lstm_output[-1]
		answer_probab = tf.matmul(lstm_answer, self.ans_sm_W) + self.ans_sm_b
		answer_probab = tf.nn.softmax(answer_probab, 'answer_probab')
		loss = tf.nn.softmax_cross_entropy_with_logits(answer_probab, answer, name = 'loss')

		return loss, answer_probab
		# loss = tf.nn.cr




