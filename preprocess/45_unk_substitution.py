#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import collections
import json

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=argparse.FileType('r'), default=sys.stdin)
parser.add_argument('--in_global_counts_file', type=argparse.FileType('r'))
parser.add_argument('--out_file', type=argparse.FileType('w'), default=sys.stdout)
parser.add_argument('--vocab_size', type=int, default=40000)
parser.add_argument('--max_unk_ratio', type=float, default=0.3)
args = parser.parse_args()

# From https://github.com/cdg720/emnlp2016/blob/master/utils.py#L322
def unkify(ws):
  unk = 'UNK'
  if ws[0].isupper():
    unk = 'C_' + unk
  if ws[0].isdigit() and ws[-1].isdigit():
    unk += '_NUMBER'
  elif len(ws) <= 3:
    pass
  elif ws.endswith('ing'):
    unk += '_ING'
  elif ws.endswith('ed'):
    unk += '_ED'
  elif ws.endswith('ly'):
    unk += '_LY'
  elif ws.endswith('s'):
    unk += '_S'
  elif ws.endswith('est'):
    unk += '_EST'
  elif ws.endswith('er'):
    unk += '_ER'
  elif ws.endswith('ion'):
    unk += '_ION'
  elif ws.endswith('ory'):
    unk += '_ORY'
  elif ws[:2] == 'un':
    unk = 'UN_' + unk
  elif ws.endswith('al'):
    unk += '_AL'
  elif '-' in ws:
    unk += '_DASH'
  elif '.' in ws:
    unk += '_DOT'
  return unk

def main():
  word_counts = json.load(args.in_global_counts_file)
  word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
  del word_counts[args.vocab_size:]
  word_counts = dict(word_counts)
  total_lines = 0
  dropped_unk_lines = 0
  for line in args.in_file:
    tokens = line.strip().split('\t')
    if len(tokens) < 2:
      continue
    tokenized = tokens[1].split()
    out_str = []
    unk_count = 0
    for token in tokenized:
      if token in word_counts:
        out_str.append(token)
      else:
        out_str.append(unkify(token))
        unk_count += 1
    total_lines += 1
    if tokenized and (unk_count / len(tokenized)) <= args.max_unk_ratio:
      args.out_file.write("\t".join([tokens[0], " ".join(out_str)] + tokens[2:]))
      args.out_file.write("\n")
    else:
      dropped_unk_lines += 1
  print("Dropped {} lines out of {}".format(dropped_unk_lines, total_lines))

if __name__ == '__main__':
  main()

