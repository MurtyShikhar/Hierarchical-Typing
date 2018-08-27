import argparse
import codecs
import gzip
import sys
import random

'''
Filter processed wiki sentences to only contain a predefined list of entities
subsample frequent entities
'''

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--wiki_file', required=True, help='directory full of wiki.gz sentence files')
parser.add_argument('-c', '--cross_wiki', required=True, help='cross wikis file')
parser.add_argument('-o', '--out_file', required=True, help='out file')
parser.add_argument('-p', '--count_file', required=True, help='entity count map for subsampling')
parser.add_argument('-k', '--K', required=True, help='export approximately at most k sentences per entity')
args = parser.parse_args()

print('Reading in cross wiki from %s.' % args.cross_wiki)
with gzip.open(args.cross_wiki, 'r') as in_f:
    entities = set(['e_slug_%s_@en' % l.strip().split('\t')[1] for l in in_f])

print('Reading in count files for subsampling from %s.' % args.count_file)
with open(args.count_file, 'r') as in_f:
    parts = [l.strip().split('\t') for l in in_f]
    entity_prob = {entity: min(int(args.K)/float(count), 1.0) for count, entity in parts}

matches = 0
print('Reading in and filtering sentences from %s to %s.' % (args.wiki_file, args.out_file))
with gzip.open(args.out_file, 'w') as out_f:
    with gzip.open(args.wiki_file, 'r') as in_f:
        for line_num, line in enumerate(in_f):
            sys.stdout.write('\rProcessing line %dK. Matches so far - %d' % (line_num / 1000, matches))
            sys.stdout.flush()
            wiki_page, _, entity, prev_sentence, current_sentence, next_sentence = line.split('\t')
            if entity in entities:
                if entity not in entity_prob or random.uniform(0, 1) < entity_prob[entity]:
                    matches += 1
                    out_f.write(line)
print ('\nDone. Matched %d examples.' % matches)
