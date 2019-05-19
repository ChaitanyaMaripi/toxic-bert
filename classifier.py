""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-19 17:29:15
"""

# Load packages
import os
import random
import logging
import argparse
import collections
import numpy as np
import tensorflow as tf
import tokenization
import optimization
from optimization import create_optimizer
import modeling
from modeling import BertConfig
from processor import ToxicProcessor

# Setup loggin configuration
logging.basicConfig(
	format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
	datefmt = "%m/%d/%Y %H:%M:%S",
	level = logging.INFO,
	filename="training.log", filemode="w"
)

# Access logging instance
logger = logging.getLogger(__name__)

# Argument parser instance
parser = argparse.ArgumentParser()
# Argument access instance
args = parser.parse_args()


# Arguments Expected
parser.add_argument(
	"--data_dir",
	type = str,
	required = True
)
parser.add_argument(
	"--vocab_file",
	type = str,
	required = True,
)
parser.add_argument(
	"--bert_config_file",
	type = str,
	required = True,
)
parser.add_argument(
	"--init_checkpoint",
	type = str,
	required = True,
)
parser.add_argument(
	"--output_dir",
	type = str,
	required = True,
)
parser.add_argument(
	"--model_dir",
	type = str,
	required = True,
)
parser.add_argument(
	"--do_train",
	action = "store_true"
)
parser.add_argument(
	"--do_eval",
	action = "store_true"
)
parser.add_argument(
	"--do_predict",
	action = "store_true"
)
parser.add_argument(
	"--do_lower_case",
	action = "store_true"
)
parser.add_argument(
	"--max_seq_length",
	default = 256,
	type = int
)
parser.add_argument(
	"--train_batch_size",
	default = 3,
	type = int
)
parser.add_argument(
	"--eval_batch_size",
	default = 12,
	type = int
)
parser.add_argument(
	"--predict_batch_size",
	default = 32,
	type = int
)
parser.add_argument(
	"--learning_rate",
	default = 2e-5,
	type = float
)
parser.add_argument(
	"--num_train_epochs",
	default = 4.0,
	type = float
)
parser.add_argument(
	"--warmup_proportion",
	default = 0.1,
	type = float
)
parser.add_argument(
	"--seed",
	type = int,
	default = 69
)


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.is_real_example = is_real_example


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
	"""Converts a single `InputExample` into a single `InputFeatures`."""
	
	label_map = dict()
	# Create label mapping
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	# Access tokens from text_a
	tokens_a = tokenizer.tokenize(example.text_a)
	
	# If text_b is present, access text_b
	tokens_b = None
	if example.text_b:
		tokens_b = tokenizer.tokenize(example.text_b)

	if tokens_b:
		# Modifies `tokens_a` and `tokens_b` in place so that the total
		# length is less than the specified length.
		# Account for [CLS], [SEP], [SEP] with "- 3"
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
	else:
		# Account for [CLS] and [SEP] with "- 2"
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]

	# The convention in BERT is:
    # For single sequences:
	#   tokens:   [CLS] the dog is hairy . [SEP]
	#   type_ids: 0     0   0   0  0     0 0
	#
	# Where "type_ids" are used to indicate whether this is the first
	# sequence or the second sequence. The embedding vectors for `type=0` and
	# `type=1` were learned during pre-training and are added to the wordpiece
	# embedding vector (and position vector). This is not *strictly* necessary
	# since the [SEP] token unambiguously separates the sequences, but it makes
	# it easier for the model to learn the concept of sequences.
	#
	# For classification tasks, the first vector (corresponding to [CLS]) is
	# used as the "sentence vector". Note that this only makes sense because
	# the entire model is fine-tuned.
	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)

	if tokens_b:
		for token in tokens_b:
			tokens.append(token)
			segment_ids.append(1)
		# Add separator token
		tokens.append("[SEP]")
		segment_ids.append(1)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	# Access label id of the current example
	label_id = label_map[example.label]

	# Return created feature
	feature = InputFeatures(
		input_ids = input_ids,
		input_mask = input_mask,
		segment_ids = segment_ids,
		label_id = label_id,
		is_real_example = True
	)
	return feature


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
	"""Convert a set of `InputExample`s to a list of `InputFeatures`."""
	features = list()
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			try:
				logger.info("Writing example %d of %d" % (ex_index, len(examples)))
			except Exception as e:
				print(e)
		# Create feature out of a single sample
		feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
		# Store created features
		features.append(feature)
	return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) >= len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
	"""Creates a classification model using BERT."""

	model = modeling.BertModel(
		config = bert_config,
		is_training = is_training,
		input_ids = input_ids,
		input_mask = input_mask,
		token_type_ids = segment_ids,
		use_one_hot_embeddings = False
	)
	# If you want to use the token-level output, use model.get_sequence_output()
	output_layer = model.get_pooled_output()

	# Get hidden size
	hidden_size = output_layer.shape[-1].value

	# Weights
	output_weights = tf.get_variable(
		"weights",
		[num_labels, hidden_size],
		initializer = tf.contrib.layers.xavier_initializer()
	)
	# Bias
	output_bias = tf.get_variable("bias", [num_labels], initializer=tf.zeros_initializer())
	
	# Set loss
	with tf.variable_scope("loss"):
		if is_training: 
            output_layer = tf.nn.dropout(output_layer, rate=0.1)
		# Linear transformation
		logits = tf.nn.bias_add(tf.matmul(output_layer, output_weights, transpose_b=True), output_bias)
		# Softmax layer
		probabilities = tf.nn.softmax(logits, axis=-1)
		# # Log-softmax layer
		log_probs = tf.nn.log_softmax(logits, axis=-1)
		# One hot labels
		one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
		# Example per loss
		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		# Reduce loss
		loss = tf.reduce_mean(per_example_loss)
	return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps):
	"""Returns `model_fn` closure for Estimator."""

	def model_fn(features, labels, mode, params):
		"""The `model_fn` for Estimator."""
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		label_ids = features["label_ids"]

		is_real_example = None
		if "is_real_example" in features:
			is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
		else:
			is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

		is_training = (mode == tf.estimator.ModeKeys.TRAIN)

		# Create Model
		(total_loss, per_example_loss, logits, probabilities) = create_model(bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels)
		
		tvars = tf.trainable_variables()
		initialized_variable_names = dict()
		if init_checkpoint:
			(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
		
		# Train from initial checkpoint
		tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
		
		# Print pre-trained layers
		for var in tvars:
			init_string = ""
			if var.name in initialized_variable_names:
				init_string = ", ---INIT_FROM_CKPT---"
			logger.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

		output_spec = None
		if mode == tf.estimator.ModeKeys.TRAIN:
			train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)
			logging_hook = tf.train.LoggingTensorHook({"loss" : total_loss}, every_n_iter=100)

			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode = mode,
				loss = total_loss,
				train_op = train_op,
				training_hooks = [logging_hook],
				scaffold_fn = None
			)
		elif mode == tf.estimator.ModeKeys.EVAL:
			def metric_fn(per_example_loss, label_ids, logits, is_real_example):
				predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
				accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions, weights=is_real_example)
				val_loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
				return {
					"eval_accuracy": accuracy,
					"eval_loss": val_loss,
				}
			eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, is_real_example])
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode = mode,
				loss = total_loss,
				eval_metrics = eval_metrics,
				scaffold_fn = None
			)
		else:
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode = mode,
				predictions = {"probabilities": probabilities},
				scaffold_fn = None
			)
		return output_spec
	return model_fn


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
	"""Convert a set of `InputExample`s to a TFRecord file."""

	writer = tf.python_io.TFRecordWriter(output_file)

	for (ex_index, example) in enumerate(examples):
		if ex_index % 100 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		# Get feature for the single example
		feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

		def create_int_feature(values):
			return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

		# Get Int features
		features = collections.OrderedDict()
		features["input_ids"] = create_int_feature(feature.input_ids)
		features["input_mask"] = create_int_feature(feature.input_mask)
		features["segment_ids"] = create_int_feature(feature.segment_ids)
		features["label_ids"] = create_int_feature([feature.label_id])
		features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

		# Create tf record file, this goes into estimator while running
		tf_example = tf.train.Example(features=tf.train.Features(feature=features))

		# Write trainable example to file
		writer.write(tf_example.SerializeToString())
	writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""

	name_to_features = {
		"input_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"input_mask": tf.FixedLenFeature([seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"label_ids": tf.FixedLenFeature([], tf.int64),
		"is_real_example": tf.FixedLenFeature([], tf.int64),
	}

	def _decode_record(record, name_to_features):
		"""Decodes a record to a TensorFlow example."""
		example = tf.parse_single_example(record, name_to_features)

		# tf.Example only supports tf.int64, but the TPU only supports tf.int32.
		# So cast all int64 to int32.
		for name in list(example.keys()):
			t = example[name]
			if t.dtype == tf.int64:
				t = tf.to_int32(t)
			example[name] = t
		return example

	def input_fn(params):
		"""The actual input function."""
		batch_size = params["batch_size"]
		# For training, we want a lot of parallel reading and shuffling.
		# For eval, we want no shuffling and parallel reading doesn't matter.
		d = tf.data.TFRecordDataset(input_file)
		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=100)

		d = d.apply(
			tf.contrib.data.map_and_batch(
				lambda record: _decode_record(record, name_to_features),
				batch_size = batch_size,
				drop_remainder = drop_remainder
			)
		)
		return d
	return input_fn


def main(_):
	# Set random seed
	random.seed(args.seed)

	# Create model and output directories
	tf.gfile.MakeDirs(args.output_dir)
	tf.gfile.MakeDirs(args.model_dir)

	# Data processor object
	processor = ToxicProcessor()

	# Get pre-trained bert configuration
	logger.info("---Getting BERT config from {}---".format(args.bert_config_file))
	bert_config = BertConfig.from_json_file(args.bert_config_file)

	# Raise error if max sequence length of examples are greater than the bert pre-trained max sequence length
	if args.max_seq_length > bert_config.max_position_embeddings:
		raise ValueError("equence length {} greater than maximum sequence length %d".format(args.max_seq_length, bert_config.max_position_embeddings))

	# Configure tokenizer
	logger.info("{:-^50s}".format("---Initiating Tokenizer---"))
	
	# Vaidate tokenization configuration
	tokenization.validate_case_matches_checkpoint(args.do_lower_case, args.init_checkpoint)
	tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

	logger.info("{:-^50s}".format("---Setting Estimator Run Configuration---"))
	tpu_cluster_resolver = None
	is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
	
	train_examples = None
	num_train_steps = None
	num_warmup_steps = None
	
	if args.do_train:
		# Get the training sampels
		train_examples = processor.get_train_examples(args.file_name, args.data_dir)
		# Calculate training steps
		num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)
		num_warmup_steps = int(num_train_steps * args.warmup_proportion)
		# Save model weights after each epoch
		save_after_steps = num_train_steps / args.num_train_epochs

	# Set TPU run cofiguration
	run_config = tf.contrib.tpu.RunConfig(
		cluster=tpu_cluster_resolver,
		master=None,
		model_dir=args.model_dir,
		save_checkpoints_steps=save_after_steps,
		tpu_config=tf.contrib.tpu.TPUConfig(
			iterations_per_loop=100,
			num_shards=4,
			per_host_input_for_training=is_per_host
		)
	)
	
	# Get labels
	logger.info("{:-^50s}".format("---Getting Labels---"))
	label_list = processor.get_labels()
	num_labels = len(label_list)

	logger.info("{:-^50s}".format("---Building Model---"))
	
	# Build model using the set configuration
	model_fn = model_fn_builder(
		bert_config = bert_config,
		num_labels = num_labels,
		init_checkpoint = args.init_checkpoint,
		learning_rate = args.learning_rate,
		num_train_steps = num_train_steps,
		num_warmup_steps = num_warmup_steps
	)

	# If TPU is not available, this will fall back to normal Estimator on CPU or GPU
	estimator = tf.contrib.tpu.TPUEstimator(
		use_tpu = False,
		model_fn = model_fn,
		config = run_config,
		train_batch_size = args.train_batch_size,
		eval_batch_size = args.eval_batch_size,
		predict_batch_size = args.predict_batch_size
	)

	logger.info("{:-^50s}".format("---Confguring Early Stopping---"))
	
	# Define early stopping criteria
	early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
		estimator,
		metric_name="loss",
		max_steps_without_decrease=10,
		min_steps=100
	)

	if args.do_train:
		logging.info("{:-^50s}".format("---Training Under Progress---"))
		
		train_file = os.path.join(args.output_dir, "train.tf_record")

		# Convert train samples into its corresponding features
		file_based_convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, train_file)

		logger.info("{:-^50s}".format("---Training---"))
		logger.info("Num examples = %d", len(train_examples))
		logger.info("Batch size = %d", args.train_batch_size)
		logger.info("Num steps = %d", num_train_steps)

		# Build input function
		train_input_fn = file_based_input_fn_builder(
			input_file=train_file,
			seq_length=args.max_seq_length,
			is_training=True,
			drop_remainder=True
		)
		# Fine-tune BERT
		estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[early_stopping])

	if args.do_eval:
		logging.info("{:-^50s}".format("---Evaluation Under Progress---"))

		# Get exampls for validation
		eval_examples = processor.get_dev_examples(args.data_dir)
		num_actual_eval_examples = len(eval_examples)

		dev_file = os.path.join(args.output_dir, "dev.tf_record")

		file_based_convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, dev_file)

		# This tells the estimator to run through the entire set.
		eval_steps = None
		eval_drop_remainder = False
		eval_input_fn = file_based_input_fn_builder(
			input_file=eval_file,
			seq_length=args.max_seq_length,
			is_training=False,
			drop_remainder=eval_drop_remainder
		)

		result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

		output_eval_file = os.path.join(args.output_dir, "dev_results.txt")
		with tf.gfile.GFile(output_eval_file, mode="w") as writer:
			logger.info("---Eval_Results---")
			for key in sorted(result.keys()):
				tf.logging.info("  %s = %s", key, str(result[key]))
				writer.write("%s = %s\n" % (key, str(result[key])))

	if args.do_predict:
		logging.info("{:-^50s}".format("---Prediction Under Progress---"))

		# Get examples for prediction
		predict_examples = processor.get_test_examples(args.file_name, args.data_dir)
		num_actual_predict_examples = len(predict_examples)

		# Set prediction file
		predict_file = os.path.join(args.output_dir, "test.tf_record")
		
		# Convert examples into features
		file_based_convert_examples_to_features(predict_examples, label_list, args.max_seq_length, tokenizer, predict_file)

		# Logging
		logging.info("{:-^50s}".format("Prediction"))
		logging.info("Num examples = %d (%d actual, %d padding)", len(predict_examples), num_actual_predict_examples, (len(predict_examples) - num_actual_predict_examples))
		logging.info("Batch size = %d", args.predict_batch_size)

		# Create prediction function
		predict_input_fn = file_based_input_fn_builder(
			input_file = predict_file,
			seq_length = args.max_seq_length,
			is_training = False,
			drop_remainder = False
		)
		# Perform prediction and get results
		result = estimator.predict(input_fn=predict_input_fn)

		# Set output file
		output_predict_file = os.path.join(args.output_dir, "test_prediction.csv")
		
		# Write results one line at a time
		with tf.gfile.GFile(output_predict_file, mode="w") as writer:
			logging.info("{:-^50s}".format("---Saving Prediction Results---"))
			for (i, prediction) in enumerate(result):
				probabilities = prediction["probabilities"]
				output_line = "\t".join(str(class_probability) for class_probability in probabilities) + "\n"
				writer.write(output_line)


if __name__ == "__main__":
	logger.info("{:-^50s}".format("Starting: ToxicBERT"))
	tf.app.run()
	logger.info("{:-^50s}".format("Run-Completed"))