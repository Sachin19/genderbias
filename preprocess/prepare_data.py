import sys
import os
import hashlib
import struct
import subprocess
import collections
import random
import tensorflow as tf
from tensorflow.core.example import example_pb2


# dm_single_close_quote = u'\u2019' # unicode
# dm_double_close_quote = u'\u201d'
# END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 200000
CHUNK_SIZE = 10000 # num examples per chunk, for the chunked data

def chunk_file(set_name, chunks_dir):
  in_file = os.path.join(finished_files_dir, "%s.bin" % set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all(chunks_dir):
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'test', "dev"]:
    print "Splitting %s data into chunks..." % set_name
    chunk_file(set_name, chunks_dir)
  print "Saved chunked data in %s" % chunks_dir

def create_keyword_labelset(fn1, fn2, fn3):
  f1 = open(fn1)
  f2 = open(fn2)
  f3 = open(fn3)
  keywords = {}
  c = 0

  for l in f1:
    items = l.strip().split("\t")
    for item in items[1:]:
      if item not in keywords:
        keywords[item]=c
        c+=1

  for l in f2:
    items = l.strip().split("\t")
    for item in items[1:]:
      if item not in keywords:
        keywords[item]=c
        c+=1

  for l in f3:
    items = l.strip().split("\t")
    for item in items[1:]:
      if item not in keywords:
        keywords[item]=c
        c+=1

  print "number of keywords",c, len(keywords)
  return keywords

def write_to_bin(tokenized_file, out_file, shuffle=False, makevocab=False, keywords={}, label_file=None, topic_file=None):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

  if makevocab:
    vocab_counter = collections.Counter()
  lines = open(tokenized_file).readlines()
  if label_file:
    labelss = open(label_file).readlines()
  if topic_file:
    topicss = open(topic_file).readlines()
  indexed_lines = list(enumerate(lines))
  if shuffle:
    random.shuffle(indexed_lines)
  c=0
  with open(out_file, 'wb') as writer:
    for idx, text in indexed_lines:
      topics = None
      kw = None
      if label_file and topic_file:
        labels = labelss[idx].strip().split("\t")
        kw = ['0' for i in range(len(keywords))]
        for k in labels[1:]:
          kw[keywords[k]] = '1'
        kw = " ".join(kw)
        label = labels[0]
        topics = topicss[idx].strip()
        comment = text.strip()
      elif label_file:
        labels = labelss[idx].strip().split("\t")
        kw = ['0' for i in range(len(keywords))]
        for k in labels[1:]:
          kw[keywords[k]] = '1'
        kw = " ".join(kw)
        label = labels[0]
        comment = text.strip()
      else:
        try:
          label, comment = text.strip().split("\t")
        except Exception as e:
          continue

      if comment == "" or label == "":
        print "WARNING: comment or label is empty, Skipping..."
        continue
      if c % 10000 == 0:
        print "Writing sentence %i of %i; %.2f percent done" % (c, len(lines), float(c)*100.0/float(len(lines)))
      c+=1
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['comment'].bytes_list.value.extend([comment])
      tf_example.features.feature['label'].bytes_list.value.extend([label])
      if kw:
        tf_example.features.feature['keywords'].bytes_list.value.extend([kw])
      if topics:
        tf_example.features.feature['topics'].bytes_list.value.extend([topics])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        comment_tokens = comment.split()
        
        comment_tokens = [t for t in comment_tokens if t.strip() not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        comment_tokens = [t.strip() for t in comment_tokens]
        comment_tokens = [t for t in comment_tokens if t!=""]

        # tokens = art_tokens + abs_tokens
        # tokens = [t.strip() for t in tokens] # strip
        # tokens = [t for t in tokens if t!=""] # remove empty
        if "</s>" in comment_tokens or "</s>" in comment_tokens:
          print comment, label
          break
        vocab_counter.update(comment_tokens)

  print "Finished writing file %s\n" % out_file

  # write vocab to file
  if makevocab:
    print "Writing vocab file..."
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')

    print "Finished writing vocab file"


if __name__ == '__main__':
  if len(sys.argv) != 6:
    print "USAGE: python make_datafiles.py <labeltype> <train_file> <dev_file> <test_file> <save_dir>"
    sys.exit()

  labeltype = sys.argv[1]
  if labeltype == "0": #label is included in the traintext file
    train_file = sys.argv[2]
    dev_file = sys.argv[3]
    test_file = sys.argv[4]
    finished_files_dir = sys.argv[5]

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
    chunks_dir = os.path.join(finished_files_dir, "chunked")

    write_to_bin(dev_file, os.path.join(finished_files_dir, "dev.bin"))
    write_to_bin(test_file, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(train_file, os.path.join(finished_files_dir, "train.bin"), makevocab=True)
  elif labeltype == "1": #first label is gender, others are keywords
    train_text = sys.argv[2]+"_text_topicalremoved.txt"
    train_labels = sys.argv[2]+"_labels.txt"
    dev_text = sys.argv[3]+"_text_topicalremoved.txt"
    dev_labels = sys.argv[3]+"_labels.txt"
    test_text = sys.argv[4]+"_text_topicalremoved.txt"
    test_labels = sys.argv[4]+"_labels.txt"
    finished_files_dir = sys.argv[5]
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
    chunks_dir = os.path.join(finished_files_dir, "chunked")

    #keyword to id
    keywords = create_keyword_labelset(train_labels, test_labels, dev_labels)
    write_to_bin(dev_text, os.path.join(finished_files_dir, "dev.bin"), keywords=keywords, label_file=dev_labels)
    write_to_bin(test_text, os.path.join(finished_files_dir, "test.bin"), keywords=keywords, label_file=test_labels)
    write_to_bin(train_text, os.path.join(finished_files_dir, "train.bin"), shuffle=True, makevocab=True, keywords=keywords, label_file=train_labels)

  elif labeltype == "2":
    train_text = sys.argv[2]+"_text.txt"
    train_labels = sys.argv[2]+"_labels.txt"
    train_topics = sys.argv[2]+"_text.txt.lda"
    dev_text = sys.argv[3]+"_text.txt"
    dev_labels = sys.argv[3]+"_labels.txt"
    dev_topics = sys.argv[3]+"_text.txt.lda"
    test_text = sys.argv[4]+"_text.txt"
    test_labels = sys.argv[4]+"_labels.txt"
    test_topics = sys.argv[4]+"_text.txt.lda"
    finished_files_dir = sys.argv[5]
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
    chunks_dir = os.path.join(finished_files_dir, "chunked")

    #keyword to id
    keywords = create_keyword_labelset(train_labels, test_labels, dev_labels)
    write_to_bin(dev_text, os.path.join(finished_files_dir, "dev.bin"), keywords=keywords, label_file=dev_labels, topic_file=dev_topics)
    write_to_bin(test_text, os.path.join(finished_files_dir, "test.bin"), keywords=keywords, label_file=test_labels, topic_file=test_topics)
    write_to_bin(train_text, os.path.join(finished_files_dir, "train.bin"), shuffle=True, makevocab=True, keywords=keywords, label_file=train_labels, topic_file=train_topics)

  else:
    raise ValueError("labeltype can only be in [0|1|2]")
  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all(chunks_dir)
