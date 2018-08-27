import glob
import argparse
import gzip
from collections import defaultdict as ddict
from tqdm import tqdm
import joblib
import codecs


def read_aida(filename):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in open(filename, 'r'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


if __name__ == '__main__':
    sentences_train = read_aida('/iesl/canvas/smurty/epiKB/segmenter_data/eng.train')
    sentences_dev   = read_aida('/iesl/canvas/smurty/epiKB/segmenter_data/eng.testa')
    sentences_test  = read_aida('/iesl/canvas/smurty/epiKB/segmenter_data/eng.testb')

    joblib.dump(sentences_train, '/iesl/canvas/smurty/epiKB/segmenter_data/AIDA_train.joblib')
    joblib.dump(sentences_dev,   '/iesl/canvas/smurty/epiKB/segmenter_data/AIDA_dev.joblib')
    joblib.dump(sentences_test,  '/iesl/canvas/smurty/epiKB/segmenter_data/AIDA_test.joblib')
