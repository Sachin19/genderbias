"""This file contains code to process data into batches"""

import Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data

FLAGS = tf.app.flags.FLAGS

class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, comment, label, keywords, topics, vocab, hps):
    """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

    Args:
      comment: source text; a string. each token is separated by a single space.
      label: male or female
      vocab: Vocabulary object
      hps: hyperparameters
    """
    self.hps = hps

    # Process the comment
    comment_words = comment.strip().split()
    if len(comment_words) > hps.max_enc_steps:
      comment_words = comment_words[:hps.max_enc_steps]
    self.enc_len = len(comment_words) # store the length after truncation but before padding
    self.enc_input = [vocab.word2id(w) for w in comment_words] # list of word ids; OOVs are represented by the id for UNK token

    self.label_id = int(label == "female")
    if keywords:
      self.keywords = [int(k) for k in keywords.split()]
    if topics:
      self.topics = [float(t) for t in topics.split()]

    self.original_comment = comment.strip()
    self.original_label = label
    self.original_keywords = keywords
    self.original_topics = topics

  def pad_encoder_input(self, max_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)

class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
    self.init_labels(example_list, hps)
    self.store_orig_strings(example_list) # store the original strings

  def init_encoder_seq(self, example_list, hps):
    """Initializes the following:
        self.enc_batch:
          numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
        self.enc_lens:
          numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
        self.enc_padding_mask:
          numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

      If hps.pointer_gen, additionally initializes the following:
        self.max_art_oovs:
          maximum number of in-article OOVs in the batch
        self.art_oovs:
          list of list of in-article OOVs (strings), for each example in the batch
        self.enc_batch_extend_vocab:
          Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
    """
    # Determine the maximum length of the encoder input sequence in this batch
    max_enc_seq_len = max([ex.enc_len for ex in example_list])

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)
    # self.enc_seq_len = np.zeros((hps.batch_size,), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      for j in xrange(ex.enc_len):
        self.enc_padding_mask[i][j] = 1

  def init_labels(self, example_list, hps):
    self.labels = np.zeros((hps.batch_size,), dtype=np.int32)
    k=False
    t=False
    self.keywords = np.zeros((hps.batch_size, 346), dtype=np.int32)
    self.topics = np.zeros((hps.batch_size, 50), dtype=np.float32)
    if example_list[0].original_keywords:
      k=True
    if example_list[0].original_topics:
      t=True

    for i, ex in enumerate(example_list):
      self.labels[i] = ex.label_id
      if k:
        self.keywords[i, :] = ex.keywords[:]
      if t:
        self.topics[i,:] = ex.topics[:]


  def store_orig_strings(self, example_list):
    """Store the original article and abstract strings in the Batch object"""
    self.original_comments = [ex.original_comment for ex in example_list] # list of lists
    self.original_labels = [ex.original_label for ex in example_list] # list of strings
    self.original_keywords = [ex.original_keywords for ex in example_list] # list of lists
    self.original_topics = [ex.original_topics for ex in example_list] # list of lists



class Batcher(object):
  """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, hps, single_pass):
    """Initialize the batcher. Start threads that process the data into batches.

    Args:
      data_path: tf.Example filepattern.
      source_vocab: Vocabulary object
      target_vocab: Vocabulary object
      hps: hyperparameters
      single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
    """
    self._data_path = data_path
    self._vocab = vocab
    self._hps = hps
    self._single_pass = single_pass
    self._total_batches_so_far = 0

    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 1 # num threads to fill example queue
      self._num_batch_q_threads = 1 # num threads to fill batch queue
      self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in xrange(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in xrange(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()


  def next_batch(self):
    """Return a Batch from the batch queue.

    If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

    Returns:
      batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
    """
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    self._total_batches_so_far += 1
    return batch

  def fill_example_queue(self):
    """Reads data from file and processes into Examples which are then placed into the example queue."""
    print self._single_pass
    input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (comment, label, keywords, topics) = input_gen.next() # read the next example from file. article and abstract are both strings.

      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      example = Example(comment, label, keywords, topics, self._vocab, self._hps) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.


  def fill_batch_queue(self):
    """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
      #if self._hps.mode != 'decode':
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
      inputs = []
      for _ in xrange(self._hps.batch_size * self._bucketing_cache_size):
        inputs.append(self._example_queue.get())
      inputs = sorted(inputs, key=lambda inp: inp.enc_len) # sort by length of encoder sequence

      # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
      batches = []
      for i in xrange(0, len(inputs), self._hps.batch_size):
        batches.append(inputs[i:i + self._hps.batch_size])
      if not self._single_pass:
        shuffle(batches)
      for b in batches:  # each b is a list of Example objects
        self._batch_queue.put(Batch(b, self._hps, self._vocab))

      # else: # beam search decode mode
      #   ex = self._example_queue.get()
      #   b = [ex for _ in xrange(self._hps.batch_size)]
      #   self._batch_queue.put(Batch(b, self._hps, self._vocab))


  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, example_generator):
    """Generates article and abstract text from tf.Example.

    Args:
      example_generator: a generator of tf.Examples from file. See data.example_generator"""
    while True:
      e = example_generator.next() # e is a tf.Example
      try:
        comment_text = e.features.feature['comment'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
        label_text = e.features.feature['label'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        if 'keywords' in e.features.feature:
          keywords = e.features.feature['keywords'].bytes_list.value[0]
        else:
          keywords = None
        if 'topics' in e.features.feature:
          topics = e.features.feature['topics'].bytes_list.value[0]
        else:
          topics = None
      except ValueError:
        tf.logging.error('Failed to get comment or label from example')
        continue
      if len(comment_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
        tf.logging.warning('Found an example with empty comment. Skipping it.')
      else:
        yield (comment_text, label_text, keywords, topics)
