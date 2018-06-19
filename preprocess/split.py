from collections import defaultdict
import sys

# talk_ids = defaultdict(set)
# f=open(sys.argv[1])
# f.readline()

# for l in f:
#     items = l.split("\t")
#     if len(items) == 6:
#         talk_ids[items[0]].add(items[4])

# male_ids = list(talk_ids['male'])
# import random
# random.shuffle(male_ids)
# test_male_ids = male_ids[-10:]
# female_ids = list(talk_ids['female'])
# random.shuffle(female_ids)
# test_female_ids = female_ids[-10:]
# dev_male_ids = male_ids[-20:-10]
# dev_female_ids = female_ids[-20:-10]

trainfile = open(sys.argv[2], "w")
devfile = open(sys.argv[3], "w")
testfile = open(sys.argv[4], "w")

# dev_ids = set(dev_male_ids + dev_female_ids)
# test_ids = set(test_male_ids + test_female_ids)

import pickle
[dev_ids, test_ids] = pickle.load(open(sys.argv[5], "rb"))

f=open(sys.argv[1])
f.readline()

for l in f:
    items = l.split("\t")
    if len(items)<6:
        continue
    words = items[1].split()
    if len(words) < 3:
    	continue
    if items[4] in test_ids:
        testfile.write(items[0]+"\t"+items[1]+"\n")
    elif items[4] in dev_ids:
        devfile.write(items[0]+"\t"+items[1]+"\n")
    else:
        trainfile.write(items[0]+"\t"+items[1]+"\n")

trainfile.close()
testfile.close()
devfile.close()