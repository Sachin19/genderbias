#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import json
import nltk
import argparse
import unicodedata

from nltk.tokenize import TweetTokenizer

TOKENIZER = TweetTokenizer(reduce_len=True)

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=argparse.FileType('r'), default=sys.stdin)
parser.add_argument('--outfile', type=argparse.FileType('w'), default=sys.stdout)

args = parser.parse_args()

# From https://gist.github.com/gruber/8891611
URL_RE = re.compile(r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\x27".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))')

NAMED_ENTITY_PLACEHOLDER = "NAME"
UNK_PLACEHOLDER = "UNK"

REPLACE_LIST = {
  "he": "THEY",
  "she": "THEY",
  "she's": "THEY'RE",
  "he's": "THEY'RE",
  "his": "THEIR",
  "her": "THEIR",
  "hers": "THEIR'S",
  "him": "THEIR",  
  "himself": "SELF",
  "herself": "SELF",
  "man": "PERSON",
  "woman": "PERSON",
  "congressman": "CONGRESSPERSON",
  "congresswoman": "CONGRESSPERSON",
  "mr": "ADDRESS",
  "miss": "ADDRESS",
  "mrs": "ADDRESS",
  "ms": "ADDRESS",
  "dame": "ADDRESS",
  "ma'am": "ADDRESS",
  "lady": "ADDRESS",
  "lord": "ADDRESS",
  "sir": "ADDRESS",
}

URL_PLACEHOLDER = "HTTP_URL"

USER_NAME_RE = re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z0-9_]+)')
USER_NAME_PLACEHOLDER = "USER_NAME"

NUMBER_RE = re.compile(r'\d+')

EYE_BROWS = r"[<>]?"
EYES = r"(?:\W[xX8]|[;:=])"
NOSE = r"[co^-]?"
HAPPY_MOUTH = r"(?:(?:3+|D+|d+|P+|p+)\W|[\)\]\>\}]+)"
TEARS = r"['`]?"
SAD_MOUTH = r"(?:[\(\[\<\{@]+|\|\|)"

EMOTICON_RE = {
  " HAPPY_FACE ": re.compile(r"(?:" + EYE_BROWS + EYES + NOSE + HAPPY_MOUTH + r"|\WB\^D+\W|☺️|:\*)|\)\)+"),
  " SAD_FACE ": re.compile(r"(?:" + EYE_BROWS + EYES + TEARS + NOSE + SAD_MOUTH + r"|\WD-':|\WD:<?|\WD8\W|\WD;\W|\WD=|\WDX\W|:-c\W|:c\W|☹️)"),
}

PLACEHOLDERS = set([USER_NAME_PLACEHOLDER, URL_PLACEHOLDER, "HAPPY_FACE", "SAD_FACE", "0",
                    NAMED_ENTITY_PLACEHOLDER, UNK_PLACEHOLDER]) | set(REPLACE_LIST.values())

def ReplaceURLs(line, url_placeholder=URL_PLACEHOLDER):
  return URL_RE.sub(url_placeholder, line)

def ReplaceUserName(line, user_name_placeholder=USER_NAME_PLACEHOLDER):
  return USER_NAME_RE.sub(user_name_placeholder, line)

def ReplaceEmoticons(line):
  for replace_str, compiled_re in EMOTICON_RE.items():
    line = compiled_re.sub(replace_str, line)
  return line
      
def ReplaceNumbers(line):
  return NUMBER_RE.sub('0', line)

def LowerCase(tokens):
  for token in tokens:
    if token in PLACEHOLDERS:
      yield token
    else:
      yield token.lower()

def RemoveNamedEntities(tokens):
  """TODO
  Simplify Names replacement if the name of the addressee is known:
  f = open("/usr1/home/ytsvetko/projects/gender_bias/data/corpora/facebook/facebook_balanced_short_anonymized.tsv", "w")
for line in open("/usr1/home/ytsvetko/projects/gender_bias/data/corpora/facebook/facebook_balanced_short.tsv"):
  tokens = line.split("\t")
  name = tokens[4].lower().split()
  for n in name:
    sentence = tokens[1].replace(" "+n+" ", " NAME ")                                                        
    sentence = tokens[1].replace(" "+n, " NAME")
    sentence = tokens[1].replace(n+" ", "NAME ")
  tokens[1] = sentence
  f.write("\t".join(tokens))
  """
  result = []

  def ProcessTreeNode(tree_node):
    if isinstance(tree_node, nltk.tree.Tree):
      if tree_node.label() == "PERSON":
        result.append(NAMED_ENTITY_PLACEHOLDER)
      else:
        for node in tree_node:
          ProcessTreeNode(node)
    elif isinstance(tree_node, tuple):
      result.append(tree_node[0])
    elif isinstance(tree_node, str):
      result.append(tree_node)
    else:
      assert False, (tree_node, type(tree_node))

  pos_tagged = nltk.pos_tag(tokens)
  ne_tree = nltk.ne_chunk(pos_tagged)
  ProcessTreeNode(ne_tree)
  return result

def ProcessReplaceList(tokens):
  for token in tokens:
    yield REPLACE_LIST.get(token, token)

def main():
  for line_num, line in enumerate(args.infile):
    tokens = unicodedata.normalize('NFKD', line.decode('utf8')).encode('ascii','ignore').strip().split('\t')
    # if len(tokens) < 2:
    #   print("Skipping line:", line)
    #   continue
    to_tokenize = tokens[0]
    filtered1 = ReplaceNumbers(ReplaceEmoticons(ReplaceUserName(ReplaceURLs(to_tokenize))))
    tokenized_sentences = []
    for sent in nltk.sent_tokenize(filtered1):
      sent_tokenized = TOKENIZER.tokenize(sent)
      filtered2 = ProcessReplaceList(LowerCase(sent_tokenized))#RemoveNamedEntities(sent_tokenized)))
      tokenized_sentences.append(" ".join(filtered2))
    tokenized = " ".join(tokenized_sentences)
    # args.outfile.write("\t".join([tokens[0], tokenized.encode("utf8")] + tokens[2:]))
    args.outfile.write(tokenized.encode("utf8"))
    args.outfile.write("\n")
    # for sent in tokenized_sentences:
    #   args.outfile.write("\t".join([tokens[0], sent.encode("utf8")] + tokens[2:]))
    #   args.outfile.write("\n")
  
if __name__ == '__main__':
  main()

