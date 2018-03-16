#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import collections
import random

parser = argparse.ArgumentParser()
parser.add_argument('--in_file')
parser.add_argument('--out_male_file', type=argparse.FileType('w'))
parser.add_argument('--out_female_file', type=argparse.FileType('w'))
parser.add_argument('--label_dim', default=0, type=int)
parser.add_argument('--sentence_dim', default=1, type=int)
args = parser.parse_args()

SEED = 1233

LABELS = set(["male", "female"])

def WriteShuffled(file_obj, file_lines, num_lines_to_write):
  random.shuffle(file_lines)
  for line in file_lines:
    if num_lines_to_write == 0:
      break
    file_obj.write(line)
    num_lines_to_write -= 1

def main():
  random.seed(SEED)
  lines = collections.defaultdict(list)
  
  seen_male = set()
  seen_female = set()
  
  for line_num, line in enumerate(open(args.in_file)):
    tokens = line.strip().split('\t')
    if len(tokens) < 2:
      continue
    label, sentence = tokens[args.label_dim], tokens[args.sentence_dim]
    if label not in LABELS:
      print("Problem with preprocessing:", line)
      continue
    if label == "male" and sentence in seen_male:
      continue
    else:
      seen_male.add(sentence)
    
    if label == "female" and sentence in seen_female:
      continue
    else:
      seen_female.add(sentence)
    
    if args.label_dim == 0:
      lines[label].append(line)
    else:
      lines[label].append(label + '\t' + '\t'.join(line.strip().split('\t')[args.label_dim+1:])+'\n')

  min_count = min([len(gender_lines) for gender_lines in lines.values()])
  print("balanced line count:", min_count)

  WriteShuffled(args.out_female_file, lines["female"], min_count)
  WriteShuffled(args.out_male_file, lines["male"], min_count)


if __name__ == '__main__':
  main()

