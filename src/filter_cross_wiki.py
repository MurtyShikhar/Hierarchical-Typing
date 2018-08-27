import argparse
import codecs
import gzip
import sys
import bz2
from collections import defaultdict
from nltk.corpus import stopwords as _stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

'''
Filter cross wikis to contain only entities which are possible candidates for aida
'''


parser = argparse.ArgumentParser()
# parser.add_argument('-w', '--wiki_dir', required=True, help='directory full of wiki.gz sentence files')
parser.add_argument('-c', '--cross_wiki', required=True, help='cross wikis file')
parser.add_argument('-m', '--mentions', required=True, help='tsv from mentions to entities')
# parser.add_argument('-t', '--type_file', required=True, help='type map')
parser.add_argument('-o', '--out_file', required=True, help='out file')
parser.add_argument('-k', '--topk', type=int, help='only keep topk most probable entities per mention')
args = parser.parse_args()


def generate_queries(_mention, stemmer, stopwords):
    tokens = word_tokenize(_mention, language='english')

    # query without stop words
    no_stop_tokens = ' '.join([w for w in tokens if w not in stopwords])
    # stem words
    stemmed = ' '.join([stemmer.stem(w) for w in tokens])
    # lowercase query
    mention_lower = _mention.lower()
    lower_tokens = word_tokenize(mention_lower, language='english')
    lower_no_stop_tokens = ' '.join([w for w in lower_tokens if w not in stopwords])
    lower_stemmed = ' '.join([stemmer.stem(w) for w in lower_tokens])
    queries = [_mention, mention_lower, no_stop_tokens, lower_no_stop_tokens, stemmed, lower_stemmed]
    queries = [q for q in queries if q]
    # remove first and last token
    if len(tokens) > 1:
        no_first = ' '.join(tokens[1:])
        no_last = ' '.join(tokens[:-1])
        queries.append(no_first)
        queries.append(no_first.lower())
        queries.append(no_last)
        queries.append(no_last.lower())

    return set(queries)

# load in all the mentions from our corpus and generate queries for our alias table
print('Reading in mention - entity file and generating queries')
with codecs.open(args.mentions, "r", "utf-8") as f:
    stemmer = PorterStemmer()
    stopwords = set(_stopwords.words('english'))
    parts = [l.strip().split('\t') for l in f]
    mention_entity_list = [(mention.decode('unicode_escape'), entity.decode('unicode_escape'))
                           for mention, entity in parts]
    # make additional lenient queries for each of the mentions
    entity_query_lists = [(entity, generate_queries(mention)) for mention, entity in mention_entity_list]
    all_mentions = set([val for entity, query_list in entity_query_lists for val in query_list])

# read in cross wikis and only keep lines with a mention in our list of queries
matches = 0
with gzip.open('%s' % args.out_file, 'w') as out_f:
    with bz2.BZ2File(args.cross_wiki, "r") as in_f:
        for line_num, line in enumerate(in_f):
            if line_num % 10000 == 0:
                sys.stdout.write('\rProcessing line %dK. Matches so far - %d' % (line_num/1000, matches))
                sys.stdout.flush()
            mention, parts = line.split('\t')
            parts = parts.split(' ', 2)
            prob, entity, _ = parts
            if mention in all_mentions:
                out_line = '%s\t%s\t%s\n' % (mention, entity, prob)
                out_f.write(out_line)
                matches += 1
print('\nDone. Found %d matches' % matches)

# filter to only keep topk  candidates per mention
if args.topk:
    print('Filtering results to top %s candidates per mention.' % args.topk)
    with gzip.open('%s_top%d.gz' % (args.out_file, args.topk), 'w') as out_f:
        mention_probs = defaultdict(list)
        with gzip.open('%s' % args.out_file, 'r') as in_f:
            parts = [l.strip().split('\t') for l in in_f]
            parts = [p for p in parts if len(p) == 3]
            for line_num, (mention, entity, prob) in enumerate(parts):
                sys.stdout.write('\rProcessing line %dK.' % (line_num / 1000))
                sys.stdout.flush()
                mention_probs[mention].append((entity, float(prob)))

        for mention, prob_list in mention_probs.iteritems():
            prob_list.sort(key=lambda tup: tup[1], reverse=True)
            for entity, prob in prob_list[:args.topk]:
                out_line = '%s\t%s\t%1.8f\n' % (mention, entity, prob)
                out_f.write(out_line)
    print('\nDone. Exported top%d candidates to %s' % (args.topk, ('%s_top%s' % (args.out_file, args.topk))))
