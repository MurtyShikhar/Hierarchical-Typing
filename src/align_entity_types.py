import argparse
import gzip
import sys
import urllib

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type_map', required=True, help='freebase type mapping [freebase_id  \t types]')
parser.add_argument('-f', '--freebase_map', required=True, help='freebase wiki mapping [freebase_id  \t wiki title]')
parser.add_argument('-o', '--out_file', required=True, help='out file')
parser.add_argument('-e', '--entity_file', required=True, help='file containing entities to keep. crosswiki file works')
parser.add_argument('-m', '--figer_map', help='optional file mapping fb types to figer types')
parser.add_argument('-r', '--redirects',  help='2 col file containing wiki redirects with gross slugs')
args = parser.parse_args()

print('Reading in entities from %s.' % args.entity_file)
in_f = gzip.open(args.entity_file, 'r') if args.entity_file.endswith('.gz') else open(args.entity_file, 'r')
entities = set(['%s' % l.strip().split('\t')[1] for l in in_f])
in_f.close()

if args.redirects:
    print('resolving entities to redirects with %s' % args.redirects)
    # slug entities
    initial_count = len(entities)
    entities = set(['e_slug_%s_@en' % e for e in entities])
    with open(args.redirects, 'r') as in_f:
        entities = set([l.strip().split('\t')[1] for l in in_f if l.strip().split('\t')[0] in entities])
        # unslug
        entities = set([e.replace('e_slug_', '').replace('_@en', '') for e in entities])
    print('Mapped %d initial entities to %d resolved redirect entities.' % (initial_count, len(entities)))

print('Reading in freebase -> wiki mapping from %s.' % args.freebase_map)
with gzip.open(args.freebase_map, 'r') as in_f:
    parts = [l.strip().split('\t') for l in in_f]
    parts = [(p[0].strip(), p[1].strip()) for p in parts if len(p) == 2]
    fb_wiki_map = {'/%s' % fb_id: wiki_id for fb_id, wiki_id in parts if wiki_id in entities}

if args.figer_map:
    print('Reading in freebase -> figer mapping from from %s.' % args.figer_map)
    with open(args.figer_map, 'r') as in_f:
        parts = [l.strip().split('\t') for l in in_f]
        fb_figer_map = {fb_type: figer_type for fb_type, figer_type in parts}

print('Exporting type map form %s to %s.' % (args.type_map, args.out_file))
with gzip.open(args.type_map, 'r') as in_f:
    with gzip.open(args.out_file, 'w') as out_f:
        matches = 0
        for line_num, line in enumerate(in_f):
            sys.stdout.write('\rProcessing line %dK. Matches so far - %d' % (line_num / 1000, matches))
            sys.stdout.flush()
            line = urllib.unquote(line).decode('utf8')
            parts = line.strip().split('\t')
            if len(parts) == 2:
                fb_id, types = parts
                if fb_id in fb_wiki_map:
                    matches += 1
                    if fb_figer_map:
                        types = [fb_figer_map[fb_type] for fb_type in types.split(',') if fb_type in fb_figer_map]
                        types = ','.join(types)
                    if types:
                        out_line = '%s\t%s\t%s\n' % (fb_id, fb_wiki_map[fb_id], types)
                        out_f.write(out_line)
print ('\nDone. Matched %d entities.' % matches)
