import tensorflow as tf
import time
from flip_gradients import flip_gradient

FLAGS = tf.app.flags.FLAGS

class ClassificationModel(object):
	"""A class to model sequence to binary label (gender)"""

	def __init__(self, hps, vocab):
		self._hps = hps
		self._vocab = vocab

		self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=None)
		self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
		self.xavier_init = tf.contrib.layers.xavier_initializer()

	def get_rnn_cell(self, num_layers=1):

		cells = []
		for i in range(num_layers):
		  cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, state_is_tuple=True, initializer=self.trunc_norm_init, activation=tf.nn.softsign)
		  cell = tf.contrib.rnn.DropoutWrapper(
		      cell=cell,
		      input_keep_prob=self._dropout_input_keep_prob,
		      output_keep_prob=self._dropout_output_keep_prob)
		  cells.append(cell)

		  if len(cells) > 1:
		    final_cell = tf.contrib.rnn.MultiRNNCell(cells)
		  else:
		    final_cell = cells[0]
		return final_cell

	def _add_placeholders(self):
		"""Add placeholders to the graph. These are entry points for any input data."""
		hps = self._hps

		# encoder part
		self._batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='batch')
		self._lens = tf.placeholder(tf.int32, [hps.batch_size], name='lens')
		self._padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='padding_mask')

		#dropouts
		self._dropout_input_keep_prob = tf.placeholder(tf.float32, (), name="dropout_input_keep_prob")
		self._dropout_output_keep_prob = tf.placeholder(tf.float32, (), name="dropout_output_keep_prob")

		#labels
		self._labels = tf.placeholder(tf.int32, (hps.batch_size,), name="labels")
		self._keywords = tf.placeholder(tf.int32, (hps.batch_size, 346), name='keywords')
		self._topics = tf.placeholder(tf.float32, (hps.batch_size, 50), name='topics')

		#training
		self._lr = tf.placeholder(tf.float32, shape=(), name="sgd_learning_rate")

	def _add_encoder(self, encoder_inputs, seq_len, num_layers=1):
		"""Add a bidirectional LSTM encoder to the graph.

		Args:
		  encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
		  seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].
		  num_layers (optional). Number of forward and backward layers (defaults to 1, that is one fwd and one bwd layer)

		Returns:
		  encoder_outputs:
		    A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
		  fw_state, bw_state:
		    Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
		"""
		with tf.variable_scope('encoder'):
		  cell_fw = self.get_rnn_cell(num_layers)
		  cell_bw = self.get_rnn_cell(num_layers)
		  (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
		  encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states

		return encoder_outputs, fw_st, bw_st

	def _add_classification_layer(self, enc_states):
		hps = self._hps
		with tf.variable_scope("prediction_layer"):
			W_a = tf.get_variable("attention_matrix", [2*hps.hidden_dim, hps.emb_dim], dtype=tf.float32, initializer=self.rand_unif_init)
			v_a = tf.get_variable("v_a", [hps.emb_dim, 1], dtype=tf.float32, initializer=self.rand_unif_init)
			attention_scores = tf.tanh(tf.reshape(tf.matmul(tf.matmul(tf.reshape(enc_states, [-1, 2*hps.hidden_dim]), W_a), v_a), [hps.batch_size, -1]))
			# print attention_scores
			attention_scores = tf.nn.softmax(attention_scores, dim=1)
			attention_scores *= self._padding_mask
			masked_sums = tf.reduce_sum(attention_scores, axis=1)
			attention_scores /= tf.reshape(masked_sums, [-1, 1])
			print attention_scores
			raw_input("see")

			self._attention_scores = attention_scores
			# print attention_scores
			context_vector = tf.reduce_sum(tf.expand_dims(attention_scores, axis=2)*enc_states, axis=1)
			# print context_vector
			logits = tf.layers.dense(context_vector, 2, use_bias=True) #apply sigmoid after this to get gender labels
			cv = flip_gradient(context_vector, 0.2)
			topic_logits = tf.layers.dense(cv, 50, use_bias=True) #apply softmax to get topical distribution
			keyword_logits = tf.layers.dense(cv, 346, use_bias=True) #apply sigmoid to get multi-labels
			##TODO add gradient reversal here.
			print logits
		return logits, keyword_logits, topic_logits

	def _add_classifier(self):
		hps = self._hps
		vsize = self._vocab.size() # size of the source vocabulary

		device = "/gpu:0"
		emb_device = "/gpu:0"

		if vsize > 50000:
			emb_device = "/cpu:0"

		with tf.variable_scope('classifier'):
			# Add embedding matrices
			with tf.device(emb_device):
				with tf.variable_scope('embedding'):
					embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

  			emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._batch) # tensor with shape (batch_size, max_enc_steps, emb_size)

		with tf.device(device):
			#encoder added
			enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._lens, FLAGS.encoder_layers)
			logits, keyword_logits, topic_logits = self._add_classification_layer(enc_outputs)
			# self._keyword_probs = tf.nn.sigmoid(keyword_logits+1e-6)

			with tf.variable_scope("loss"):
				# self._gender_prob = tf.sigmoid(logits)
				# true_labels = tf.cast(self._labels, tf.float32)
				# ce_loss = -true_labels*tf.log(self._gender_prob) - (1.0-true_labels)*tf.log(1-self._gender_prob)
				ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=logits)
				# print ce_loss
				true_keywords = tf.cast(self._keywords, tf.float32)
				kw_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_keywords, logits=keyword_logits)
				print kw_loss
				# kw_loss = -true_keywords*tf.log(self._keyword_probs) - (1.0 - true_keywords)*tf.log(1-self._keyword_probs)
				kw_loss = tf.reduce_sum(kw_loss,axis=1)
				topic_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self._topics, logits=topic_logits)
				# print kw_loss
				# raw_input()
				self._loss = tf.reduce_mean(ce_loss)
				self._ce_loss = tf.reduce_mean(ce_loss)

			with tf.variable_scope("accuracy"):
				self._predictions = tf.argmax(logits, 1, output_type=tf.int32)
				# self._predictions = tf.cast(tf.greater_equal(self._gender_prob, 0.5), tf.int32)
				self._probs = tf.reduce_max(tf.nn.softmax(logits), 1)
				# self._probs = self._gender_prob
				correct_predictions = tf.cast(tf.equal(self._predictions, self._labels), tf.float32)
				self._correct_predictions = tf.reduce_sum(correct_predictions)
				self._accuracy = tf.reduce_mean(correct_predictions)

			tf.summary.scalar('loss', self._loss)
			tf.summary.scalar('ce_loss', self._ce_loss)
			tf.summary.scalar('accuracy', self._accuracy)

	def _add_train_op(self):
		"""Sets self._train_op, the op to run for training."""
		# Take gradients of the trainable variables w.r.t. the loss function to minimize
		loss_to_minimize = self._loss
		tvars = tf.trainable_variables()
		gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

		# Clip the gradients
		with tf.device("/gpu:0"):
			grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

		# Add a summary
		tf.summary.scalar('global_norm', global_norm)

		#Apply Stochastic Descent
		if FLAGS.optimizer == 'sgd':
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
		elif FLAGS.optimizer == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_lr)
		  	# embed_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_lr)
		  	# embed_optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
		elif FLAGS.optimizer == 'adagrad':
			optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
		elif FLAGS.optimizer == 'adadelta':
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.adam_lr)
		else:
			raise ValueError("Wrong optimizer parameter")

		with tf.device("/gpu:0"):
			# embed_op = embed_optimizer.apply_gradients(zip(embed_grads, embed_tvars), global_step=self.global_step, name='train_step_embed')
			# other_op = optimizer.apply_gradients(zip(other_grads, other_tvars), global_step=self.global_step, name='train_step_other')
			# self._train_op = tf.group(embed_op, other_op)
			self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

	def build_graph(self):
		"""Add the placeholders, model, global step, train_op and summaries to the graph"""
		tf.logging.info('Building graph...')
		t0 = time.time()
		self._add_placeholders()
		self._add_classifier()
		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		if self._hps.mode == 'train':
			self._add_train_op()
		self._summaries = tf.summary.merge_all()
		t1 = time.time()
		tf.logging.info('Time to build graph: %i seconds', t1 - t0)

	def _make_feed_dict(self, batch, just_enc=False):
		"""Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

		Args:
		  batch: Batch object
		  just_enc: Boolean. If True, only feed the parts needed for the encoder.
		"""
		feed_dict = {}
		feed_dict[self._batch] = batch.enc_batch
		feed_dict[self._lens] = batch.enc_lens
		feed_dict[self._padding_mask] = batch.enc_padding_mask

		if not just_enc:
		  feed_dict[self._labels] = batch.labels
		  feed_dict[self._keywords] = batch.keywords
		  feed_dict[self._topics] = batch.topics
		return feed_dict

	def run_train_step(self,sess, batch, sgd_lr):
		"""Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
		feed_dict = self._make_feed_dict(batch)
		feed_dict[self._lr] = sgd_lr
		feed_dict[self._dropout_input_keep_prob] = FLAGS.dropout_input_keep_probability
		feed_dict[self._dropout_output_keep_prob] = FLAGS.dropout_output_keep_probability

		to_return = {
		    'train_op': self._train_op,
		    'summaries': self._summaries,
		    'loss': self._loss,
		    'ce_loss':self._ce_loss,
		    'accuracy': self._accuracy,
		    'global_step': self.global_step,
		}

		return sess.run(to_return, feed_dict)

	def run_eval_step(self, sess, batch):
		"""Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
		feed_dict = self._make_feed_dict(batch)
		feed_dict[self._dropout_input_keep_prob] = 1.0 #no dropout while evaluation
		feed_dict[self._dropout_output_keep_prob] = 1.0
		to_return = {
		    'summaries': self._summaries,
		    'loss': self._loss,
		    'ce_loss': self._ce_loss,
		    'global_step': self.global_step,
		    'correct_predictions': self._correct_predictions,
		    'attention_scores': self._attention_scores,
		    'predictions': self._predictions,
		    'probs': self._probs,
		    'batch': self._batch,
		}

		return sess.run(to_return, feed_dict)



