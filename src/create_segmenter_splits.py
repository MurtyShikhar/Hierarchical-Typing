import glob
import argparse
import gzip
from collections import defaultdict as ddict
from tqdm import tqdm
import joblib


def read_entities(file_name):
    f = open(file_name)
    entities = set()
    for line in f:
        ent, _, _, _ = line.strip().split("\t")
        entities.add(ent)
    return entities

total_data_size = 0
train = []
dev = []
test = []

train_entities = read_entities("/iesl/canvas/smurty/epiKB/MIL_data/train.entities") 
dev_entities   = read_entities("/iesl/canvas/smurty/epiKB/MIL_data/dev.entities")
test_entities  = read_entities("/iesl/canvas/smurty/epiKB/MIL_data/test.entities")

print(len(dev_entities & train_entities))
print(len(test_entities & train_entities))
files = glob.glob('/iesl/canvas/pat/epiKB/typenet_data_filtered/shards/*.gz')
print("%d files found." %(len(files)))
for filename in tqdm(files, total = len(files)):
    with gzip.open(filename, mode = "r") as f:
        for line in f:
            total_data_size += 1
            #line = line.strip()
            ent = line.split("\t")[2] 
            ent = ent[7:-4]
            if ent in train_entities:
                train.append(line)
            elif ent in dev_entities:
                dev.append(line)
            else:
                test.append(line)

print("%d in train | %d in dev | %d in test" %(len(train), len(dev), len(test)))
print("Done reading files. Total data size: %d" %(total_data_size))
joblib.dump(train, '/iesl/canvas/smurty/epiKB/segmenter_data/train.joblib' )
joblib.dump(dev,   '/iesl/canvas/smurty/epiKB/segmenter_data/dev.joblib' )
joblib.dump(test, '/iesl/canvas/smurty/epiKB/segmenter_data/test.joblib' )

joblib.dump(train[:5000], '/iesl/canvas/smurty/epiKB/segmenter_data/train_small.joblib')
joblib.dump(dev[:5000],   '/iesl/canvas/smurty/epiKB/segmenter_data/dev_small.joblib' )
joblib.dump(test[:5000], '/iesl/canvas/smurty/epiKB/segmenter_data/test_small.joblib' )
