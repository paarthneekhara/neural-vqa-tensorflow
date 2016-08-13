import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle

def load_questions_answers(opts):
	
	questions = None
	answers = None
	
	t_q_json_file = join(opts.data_dir, 'MultipleChoice_mscoco_train2014_questions.json')
	t_a_json_file = join(opts.data_dir, 'mscoco_train2014_annotations.json')

	v_q_json_file = join(opts.data_dir, 'MultipleChoice_mscoco_val2014_questions.json')
	v_a_json_file = join(opts.data_dir, 'mscoco_val2014_annotations.json')
	qa_data_file = join(opts.data_dir, 'qa_data_file.pkl')
	vocab_file = join(opts.data_dir, 'vocab_file.pkl')

	# IF ALREADY EXTRACTED
	if isfile(qa_data_file):
		with open(qa_data_file) as f:
			data = pickle.load(f)
			return data

	print "Loading Training questions"
	with open(t_q_json_file) as f:
		t_questions = json.loads(f.read())
	
	print "Loading Training anwers"
	with open(t_a_json_file) as f:
		t_answers = json.loads(f.read())

	print "Loading Val questions"
	with open(v_q_json_file) as f:
		v_questions = json.loads(f.read())
	
	print "Loading Val answers"
	with open(v_a_json_file) as f:
		v_answers = json.loads(f.read())

	
	print "Ans", len(t_answers['annotations']), len(v_answers['annotations'])
	print "Qu", len(t_questions['questions']), len(v_questions['questions'])

	answers = t_answers['annotations'] + v_answers['annotations']
	questions = t_questions['questions'] + v_questions['questions']
	
	answer_vocab = make_answer_vocab(answers)
	question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
	print "Max Question Length", max_question_length
	word_regex = re.compile(r'\w+')
	training_data = []
	for i,question in enumerate( t_questions['questions']):
		ans = t_answers['annotations'][i]['multiple_choice_answer']
		if ans in answer_vocab:
			training_data.append({
				'image_id' : t_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
			question_words = re.findall(word_regex, question['question'])

			base = max_question_length - len(question_words)
			for i in range(0, len(question_words)):
				training_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]

	print "Training Data", len(training_data)
	val_data = []
	for i,question in enumerate( v_questions['questions']):
		ans = v_answers['annotations'][i]['multiple_choice_answer']
		if ans in answer_vocab:
			val_data.append({
				'image_id' : v_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
			question_words = re.findall(word_regex, question['question'])

			base = max_question_length - len(question_words)
			for i in range(0, len(question_words)):
				val_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]

	print "Validation Data", len(val_data)

	data = {
		'training' : training_data,
		'validation' : val_data,
		'answer_vocab' : answer_vocab,
		'question_vocab' : question_vocab,
		'max_question_length' : max_question_length
	}

	print "Saving qa_data"
	with open(qa_data_file, 'wb') as f:
		pickle.dump(data, f)

	with open(vocab_file, 'wb') as f:
		vocab_data = {
			'answer_vocab' : data['answer_vocab'],
			'question_vocab' : data['question_vocab'],
			'max_question_length' : data['max_question_length']
		}
		pickle.dump(vocab_data, f)

	return data

def get_question_answer_vocab(data_dir):
	vocab_file = join(data_dir, 'vocab_file.pkl')
	vocab_data = pickle.load(open(vocab_file))
	return vocab_data

def make_answer_vocab(answers):
	top_n = 1000
	answer_frequency = {} 
	for annotation in answers:
		answer = annotation['multiple_choice_answer']
		if answer in answer_frequency:
			answer_frequency[answer] += 1
		else:
			answer_frequency[answer] = 1

	answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.iteritems()]
	answer_frequency_tuples.sort()
	answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

	answer_vocab = {}
	for i, ans_freq in enumerate(answer_frequency_tuples):
		# print i, ans_freq
		ans = ans_freq[1]
		answer_vocab[ans] = i

	answer_vocab['UNK'] = top_n - 1
	return answer_vocab


def make_questions_vocab(questions, answers, answer_vocab):
	word_regex = re.compile(r'\w+')
	question_frequency = {}

	max_question_length = 0
	for i,question in enumerate(questions):
		ans = answers[i]['multiple_choice_answer']
		count = 0
		if ans in answer_vocab:
			question_words = re.findall(word_regex, question['question'])
			for qw in question_words:
				if qw in question_frequency:
					question_frequency[qw] += 1
				else:
					question_frequency[qw] = 1
				count += 1
		if count > max_question_length:
			max_question_length = count


	qw_freq_threhold = 0
	qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.iteritems()]
	# qw_tuples.sort()

	qw_vocab = {}
	for i, qw_freq in enumerate(qw_tuples):
		frequency = -qw_freq[0]
		qw = qw_freq[1]
		# print frequency, qw
		if frequency > qw_freq_threhold:
			# +1 for accounting the zero padding for batc training
			qw_vocab[qw] = i + 1
		else:
			break

	qw_vocab['UNK'] = len(qw_vocab) + 1

	return qw_vocab, max_question_length


def load_fc7_features(data_dir, split):
	import h5py
	fc7_features = None
	image_id_list = None
	with h5py.File( join( data_dir, (split + '_fc7.h5')),'r') as hf:
		fc7_features = np.array(hf.get('fc7_features'))
	with h5py.File( join( data_dir, (split + '_image_id_list.h5')),'r') as hf:
		image_id_list = np.array(hf.get('image_id_list'))
	return fc7_features, image_id_list
