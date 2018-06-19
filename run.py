from collections import namedtuple
from data import Vocab
from model import ClassificationModel
from batcher import Batcher
from tensorflow.python import debug as tf_debug
from sklearn.metrics import f1_score, precision_score, recall_score

import tensorflow as tf
import numpy as np

import util
import sys
import time
import os

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('eval_data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 300, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 300, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 128, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 10, 'max timesteps of encoder (max source text tokens in the comment)')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')

tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adam_lr', 0.0001, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('adam_epsilon', 0.0000008, 'Epsilon for Adam optimizer')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer')

tf.app.flags.DEFINE_integer('encoder_layers', 1, 'Number of layers in the encoder')

tf.app.flags.DEFINE_float('dropout_input_keep_probability', 0.8, 'Dropout probabilities')
tf.app.flags.DEFINE_float('dropout_output_keep_probability', 1.0, 'Dropout probabilities')

tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.1, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 1.0, 'for gradient clipping')

tf.app.flags.DEFINE_float('best_loss', 100000000000, 'best loss for continuing training')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

#for sgd
tf.app.flags.DEFINE_integer('non_decay_steps', 12000, 'Number of training steps for which to train without decaying learning rate. Only for SGD')
tf.app.flags.DEFINE_integer('decay_rate', 1000, 'Decay Rate')
tf.app.flags.DEFINE_float('decay_coefficient', 0.5, 'Decay Rate')
tf.app.flags.DEFINE_integer('total_steps', 20000, 'Total number of steps after which the training will be stopped')

def restore_best_model():
  """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
  tf.logging.info("Restoring bestmodel for training...")

  # Initialize all vars in the model
  sess = tf.Session(config=util.get_config())
  print "Initializing all variables..."
  sess.run(tf.initialize_all_variables())

  # Restore the best model from eval dir
  saver = tf.train.Saver([v for v in tf.all_variables() if "Adam" not in v.name and "margin_weight" not in v.name])
  print "Restoring all non-adam variables from best model in eval dir..."
  curr_ckpt = util.load_ckpt(saver, sess, "eval")
  print "Restored %s." % curr_ckpt

  # Save this model to train dir and quit
  new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
  new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
  print "Saving model to %s..." % (new_fname)
  new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
  new_saver.save(sess, new_fname)
  print "Saved."
  exit()


def setup_training(model, batcher, vocab, hps):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph
  tf.logging.info("Total number of trainable parameters: "+str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

  if FLAGS.restore_best_model:
    restore_best_model()
  saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time
  eval_saver = tf.train.Saver(max_to_keep=5)

  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                     save_model_secs=60, # checkpoint every 60 secs
                     global_step=model.global_step)
  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")
  try:
    run_training(model, batcher, sess_context_manager, sv, summary_writer, vocab, hps, eval_saver) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()


def get_eval_loss(sess, model, vocab, hps, data_path):

  eval_batcher = Batcher(data_path, vocab, hps, True)
  total_loss = 0.0
  total_ce_loss = 0.0
  total_correct_preds = 0.0
  preds = []
  truey = []
  n=0

  if FLAGS.mode == 'decode':
    pass

  while True:
    try:
      eval_batch = eval_batcher.next_batch()
      if eval_batch is None:
        break
      eval_results = model.run_eval_step(sess, eval_batch)

      batch_size = FLAGS.batch_size
      loss = eval_results['loss']
      ce_loss = eval_results['ce_loss']
      correct_predictions = eval_results['correct_predictions']
      predictions = eval_results['predictions']
      true_labels = eval_batch.labels
      preds += list(predictions)
      truey += list(true_labels)

      total_loss += loss*batch_size
      total_ce_loss += ce_loss*batch_size
      total_correct_preds += correct_predictions
      n+=batch_size
    except StopIteration:
      break

  eval_loss = total_loss/n
  eval_ce_loss = total_ce_loss/n
  accuracy = total_correct_preds/n

  print " Precision Score:", precision_score(truey, preds),
  print " Recall Score:", recall_score(truey, preds),
  print " F1 Score:", f1_score(truey, preds)
  print n

  return eval_loss, eval_ce_loss, accuracy

def run_training(model, batcher, sess_context_manager, sv, summary_writer, vocab, hps, eval_saver):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  # eval_saver = tf.train.Saver(max_to_keep=5)
  best_loss = None
  prev_loss = None
  tf.logging.info("starting run_training")

  with sess_context_manager as sess:
    if FLAGS.debug: # start the tensorflow debugger
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    train_step=0
    sgd_lr = FLAGS.lr
    tt=0.0
    counts_till_no_increase = 0
    alpha=1.0

    while True: # repeats until interrupted

      if FLAGS.optimizer == 'sgd':
        if train_step >= FLAGS.non_decay_steps and (train_step-FLAGS.non_decay_steps)%FLAGS.decay_rate == 0:
          sgd_lr *= FLAGS.decay_coefficient

        if train_step >= FLAGS.total_steps:
          tf.logging.info("Training is complete. Go home")
          break

      if train_step % 500 == 0 and train_step > 0:
        eval_loss, eval_ce_loss, accuracy = get_eval_loss(sess, model, vocab, hps, FLAGS.eval_data_path)

        if best_loss is None or accuracy > best_loss:
          tf.logging.info('Found new best model. Saving to %s', bestmodel_save_path)
          x = eval_saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
          tf.logging.info("X="+str(x))
          best_loss = accuracy
          counts_till_no_increase = 0
        else:
          counts_till_no_increase += 1

        if prev_loss is not None and FLAGS.optimizer == 'sgd':
          if prev_loss < eval_loss:
            sgd_lr *= FLAGS.decay_coefficient
        prev_loss = eval_loss

        tf.logging.info(str(train_step)+' completed, eval loss='+str(eval_loss)+' eval ce_loss='+str(eval_ce_loss)+", accuracy = "+str(accuracy)+", time perstep="+str(tt/1000))
        tt=0.0

        # if counts_till_no_increase > 80:
        #   tf.logging.info("Evaluation Loss has not improved since 40,000 steps. Terminating training!")
        #   break

      # tf.logging.info('running training step '+str(train_step)+', learning rate='+str(sgd_lr)+'...')
      batch = batcher.next_batch()
      t0=time.time()
      results = model.run_train_step(sess, batch, sgd_lr)
      t1=time.time()
      tt += (t1-t0)
      # tf.logging.info('seconds for training step: %.3f', t1-t0)

      loss = results['loss']
      accuracy = results['accuracy']
      if train_step % 100 == 0:
        tf.logging.info('Training loss and accuracy for this batch: %f, %f', loss, accuracy) # print the loss to screen

      if not np.isfinite(loss):
        raise Exception("Loss is not finite. Stopping.")

      # get the summaries and iteration number so we can write summaries to tensorboard
      summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      train_step = results['global_step'] # we need this to update our running average loss
      summary_writer.add_summary(summaries, train_step) # write the summaries
      if train_step % 100 == 0: # flush the summary writer every so often
        summary_writer.flush()


def run_eval(model, vocab, hps):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far

  _ = util.load_ckpt(saver, sess) # load a new checkpoint
  eval_loss, eval_ce_loss, accuracy = get_eval_loss(sess, model, vocab, hps, FLAGS.data_path)
  tf.logging.info('eval loss='+str(eval_loss)+', eval ce_loss='+str(eval_ce_loss)+", accuracy = "+str(accuracy))

def get_decode_results(sess, model, vocab, hps, data_path):

  eval_batcher = Batcher(data_path, vocab, hps, True)
  total_loss = 0.0
  total_correct_preds = 0.0
  predictions = np.array([])
  original_comments = []
  gold_labels = []
  attention_scores = []
  labelvalues = np.array(["male", "female"])
  predicted_labels = []
  probabilities = np.array([])

  n=0

  while True:
    try:
      eval_batch = eval_batcher.next_batch()
      if eval_batch is None:
        break
      eval_results = model.run_eval_step(sess, eval_batch)
      batch = eval_batch
      batch_size = FLAGS.batch_size
      loss = eval_results['loss']
      correct_predictions = eval_results['correct_predictions']
      predictions = eval_results['predictions']
      predicted_labels = np.concatenate((predicted_labels, labelvalues[predictions]))
      # print eval_results['probs']
      # print eval_results['batch']
      # print batch.enc_batch[0]
      # print batch.enc_batch[1]
      # print batch.enc_batch[2]
      # raw_input()
      probabilities = np.concatenate((probabilities, eval_results['probs']))
      gold_labels += batch.original_labels
      original_comments += batch.original_comments
      attention_scores += list(eval_results['attention_scores'])

      total_loss += loss*batch_size
      total_correct_preds += correct_predictions
      n+=batch_size
    except StopIteration:
      break

  eval_loss = total_loss/n
  accuracy = total_correct_preds/n

  return eval_loss, accuracy, original_comments, gold_labels, predicted_labels, attention_scores, np.array(probabilities, dtype=str)

def run_decode(model, vocab, hps):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far

  _ = util.load_ckpt(saver, sess) # load a new checkpoint
  eval_loss, accuracy, original_comments, gold_labels, predicted_labels, attention_scores, probabilities = get_decode_results(sess, model, vocab, hps, FLAGS.data_path)
  decode_dir = os.path.join(FLAGS.log_root, "decode")
  if not os.path.exists(decode_dir): os.mkdir(decode_dir)

  outputfile = open(os.path.join(decode_dir, "outputs"), "w")
  attentionfile = open(os.path.join(decode_dir, "attention_scores"), "w")
  for comment, gl, pl, prob in zip(original_comments, gold_labels, predicted_labels, probabilities):
    outputfile.write("\t".join([comment, str(gl), str(pl), str(prob)]))
    outputfile.write("\n")

  for attention_score in attention_scores:
    attentionfile.write(" ".join(np.array(attention_score, dtype=str)))
    attentionfile.write("\n")

  outputfile.close()
  attentionfile.close()

  tf.logging.info('eval loss='+str(eval_loss)+", accuracy = "+str(accuracy))

def main(unused_argv):
  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  tf.logging.info('Running the code in %s mode...', (FLAGS.mode))

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a source vocabulary

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['adam_epsilon','mode', 'loss', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_enc_steps']
  hps_dict = {}
  for key,val in FLAGS.__flags.iteritems(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # Create a batcher object that will create minibatches of data

  tf.set_random_seed(1233) # a seed value for randomness

  if hps.mode == 'train':
    print "creating model..."
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=False)
    model = ClassificationModel(hps, vocab)
    # batcher = None
    setup_training(model, batcher, vocab, hps)

  elif hps.mode == 'eval':
  	model = ClassificationModel(hps, vocab)
  	run_eval(model, vocab, hps)

  elif hps.mode == 'decode':
    model = ClassificationModel(hps, vocab)
    run_decode(model, vocab, hps)

  # elif hps.mode == 'decode':
  #   decode_model_hps = hps  # This will be the hyperparameters for the decoder model
  #   decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries


  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
  tf.app.run()
