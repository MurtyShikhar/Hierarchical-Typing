import glob
import re
import bz2
import argparse
import gzip
from collections import defaultdict as ddict
from tqdm import tqdm
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type_system', required=True, help='freebase type system')
args = parser.parse_args()

type_system = args.type_system
pattern = re.compile('[^\s\w]+')
# === TODO: create same split for both entity typing and linking
def get_figer_types(file_name):
    f = open(file_name)
    types = {"NO_TYPES" : "NO_TYPES"}

    for line in f:
        fb_type, figer_type = line.strip().split("\t")

        types[fb_type] = figer_type
        types[figer_type] = figer_type

    return types


def get_typenet_types(file_name):
    f = open(file_name)
    types = set()
    for line in f:
        e1, _, e2 = line.strip().split(" ")
        types.add(e1)
        types.add(e2)


    print("%d types found" %len(types))
    return types


def get_fb_types(file_name, type_system, figer = False):
    f = open(file_name)
    entity_type_map = {}
    for line in f:
        dat = line.strip().split("\t")
        if len(dat) < 2:
            continue

        ent, types = dat
        #if ent not in entity_set:
        #    continue

        types = types.split(",")
        types = filter(lambda _type: _type in type_system, types)

        if len(types) == 0:
            types.append("NO_TYPES")

        if figer:
            types = map(lambda _type : type_system[_type], types)

        entity_type_map[ent] = types

    return entity_type_map

def get_transitive_closure(types, transitive_closure, type2idx, idx2type):
    types = map(lambda _type : type2idx[_type], types)

    all_types = set(types)

    for _type in types:
        for parent in xrange(len(transitive_closure[_type])):
            if transitive_closure[_type][parent]:
                all_types.add(parent)

    all_types = list(all_types)
    all_types = filter(lambda id: not idx2type[id].startswith("Synset"), all_types)


    return all_types


def get_entity_type_dict(fb_type_map, file_name, figer = False):
    f = open(file_name)
    lines = f.readlines()
    entity_type_dict = {}
    entity_type_dict_orig = {}

    for line in tqdm(lines, total = len(lines)):
        ent, fb_ent, _, _ = line.strip().split("\t")

        if figer:
            types = [type2idx[_type] for _type in fb_type_map[fb_ent] ]
        else:
            types = get_transitive_closure(fb_type_map[fb_ent], transitive_closure, type2idx, idx2type)
            types_orig = [type2idx[_type] for _type in fb_type_map[fb_ent] ]
            entity_type_dict_orig[ent] = types_orig

        entity_type_dict[ent] = types

    return entity_type_dict, entity_type_dict_orig



# ========= Code for creating Wiki alias table (CrossWikis)

def getLnrm(arg, pattern):
  """Normalizes the given arg by stripping it of diacritics, lowercasing, and
  removing all non-alphanumeric characters.
  """
  arg = pattern.sub('', arg)
  arg = arg.lower()

  return arg

def get_redirects(file_path):
    '''
        Obtain entity - redirected_entity
    '''
    f = open(file_path)
    redirects = {}

    for line in f:
        e, redirect_e = line.strip().split("\t")
        redirects[e] = redirect_e

    return redirects

def read_dictionary(dictionary_file, entities, redirects_file, all_mentions):
    '''
        Read the dictionary file and return a list of (entity, probab) for every mention.
        Need to run this just once to create the shelve file
    '''

    print("======== Reading Entities ========")
    crosswikis_size = 297073139

    redirects = get_redirects(redirects_file)


    f = bz2.BZ2File(dictionary_file)
    mention_data = {}  # serves as a hashtable
    inverse_idx = {}
    lines_written = 0

    for line in tqdm(f, total = crosswikis_size):
        curr = line.strip().split("\t")
        if (len(curr) >= 2):
            curr_mention = getLnrm(curr[0], pattern) # lowercase this
            if curr_mention not in all_mentions: continue
            aux_data = curr[1].split(" ")
            probab = float(aux_data[0])
            entity = "e_slug_%s_@en" %aux_data[1]
            entity = redirects[entity] if entity in redirects else entity # find its redirect
            entity = entity[7:-4] # get rid of the e_slug

            if (entity in entities):
                if curr_mention not in mention_data:
                    # we want the list to always be sorted in reverse order
                    mention_data[curr_mention] = {}

                if entity not in inverse_idx:
                    inverse_idx[entity] = []

                lines_written += 1
                if entity not in mention_data[curr_mention]:
                    mention_data[curr_mention][entity] = probab
                else:
                    mention_data[curr_mention][entity] += probab

                inverse_idx[entity].append(curr_mention)


    f.close()
    print("total considered: %d" %lines_written)
    print("Number of mentions written : %d" %(len(mention_data)))
    

    f = open("inverse_idx.txt", "w")

    mention_data_new = {}

    for mention in mention_data:
        mention_data_new[mention] = list(mention_data[mention].iteritems())


    for ent in inverse_idx:
        all_mentions = inverse_idx[ent]
        f.write("="*50)
        f.write("\n")
        f.write("%s\n" %ent)
        for mention in all_mentions:
            f.write("%s\n" %mention)


    f.close()
    # store the list version
    return mention_data_new


files = ["/iesl/canvas/pat/epiKB/typenet_data_filtered/splits/linking_splits/train.gz"]


def create_bags(_file):
    entity_bags = ddict(list)
    total_data_size = 0
    with gzip.open(_file, mode = "r") as f:
        for line in f:
            total_data_size += 1
            ent = line.split("\t")[2]
            entity_bags[ent[7:-4]].append(line)
    print("Done reading files. Total data size: %d" %(total_data_size))
    min_bag_size = 1000
    for ent in entity_bags:
        min_bag_size = min(len(entity_bags[ent]),  min_bag_size)
    print(min_bag_size)
    return entity_bags

def get_mentions(_file):
    mentions = set()

    with gzip.open(_file, mode = "r") as f:
        for line in f:
            title, gold_mention, entity, prev_sentence, curr_sentence, next_sentence = line.split("\t")
            gold_mention = " ".join(gold_mention[9:].split("_"))
            gold_mention = getLnrm(gold_mention, pattern) # lowercase this
            mentions.add(gold_mention)
    return mentions

#files = glob.glob('/iesl/canvas/pat/epiKB/typenet_data_filtered/shards/*.gz')
#train_bags = create_bags("/iesl/canvas/pat/epiKB/typenet_data_filtered/splits/linking_splits/train.gz")
#dev_bags = create_bags("/iesl/canvas/pat/epiKB/typenet_data_filtered/splits/linking_splits/dev.gz")
#test_bags = create_bags("/iesl/canvas/pat/epiKB/typenet_data_filtered/splits/linking_splits/test.gz")

all_mentions = get_mentions("/iesl/canvas/pat/epiKB/typenet_data_filtered/splits/linking_splits/all_sentences.gz")
print("%d total mentions" %len(all_mentions))
#joblib.dump(train_bags, '/iesl/canvas/smurty/epiKB/MIL_data/entity_bags_joint_train.joblib')
#joblib.dump(dev_bags, '/iesl/canvas/smurty/epiKB/MIL_data/entity_bags_joint_dev.joblib')
#joblib.dump(test_bags, '/iesl/canvas/smurty/epiKB/MIL_data/entity_bags_joint_test.joblib')

if type_system == "typenet":
    entities = set([l.strip().split("\t")[0] for l in open("/iesl/canvas/smurty/epiKB/MIL_data/all.entities")])
    entity_dict = {ent : idx for (idx, ent) in enumerate(entities)}

    joblib.dump(entity_dict, "/iesl/canvas/smurty/epiKB/MIL_data/entity_dict.joblib")
    alias_table =  read_dictionary("/iesl/canvas/nmonath/data/crosswikis/dictionary.bz2", entity_dict, "/iesl/canvas/smurty/wiki-data/enwiki-20160920-redirect.tsv", all_mentions)
    joblib.dump(alias_table, "/iesl/canvas/smurty/epiKB/MIL_data/alias_table.joblib")

    type2idx = joblib.load("/iesl/canvas/smurty/epiKB/types_annotated/TypeNet_type2idx.joblib")
    idx2type = {idx : _type for (_type, idx) in type2idx.iteritems()}
    transitive_closure = joblib.load("/iesl/canvas/smurty/epiKB/types_annotated/TypeNet_transitive_closure.joblib")

    fb_type_map = get_fb_types("/iesl/canvas/pat/data/freebase/entity_to_fbtypes", type2idx)
    entity_type_dict, _ = get_entity_type_dict(fb_type_map, "/iesl/canvas/smurty/epiKB/MIL_data/all.entities")



else:
    #maps freebase types to figer types
    fb2figer = get_figer_types("/iesl/canvas/smurty/epiKB/AIDA_linking/figer_type.map")
    figer_types = set(fb2figer.values())
    type2idx  = {_type : idx for (idx, _type) in enumerate(figer_types)}
    idx2types = {idx : _type for (_type, idx) in type2idx.iteritems()}
    joblib.dump(type2idx, "/iesl/canvas/smurty/epiKB/MIL_data/figer_type_dict.joblib")

    fb_type_map = get_fb_types("/iesl/canvas/pat/data/freebase/entity_to_fbtypes", fb2figer, figer=True)

    entity_type_dict, entity_type_dict_orig = get_entity_type_dict(fb_type_map, "/iesl/canvas/smurty/epiKB/MIL_data/all.entities", figer = True)
    joblib.dump(entity_type_dict_orig, "/iesl/canvas/smurty/epiKB/MIL_data/entity_%s_type_dict_orig.joblib" %(type_system))


print("Created type dict from entities to transitive closure of typenet types, indexified.")
joblib.dump(entity_type_dict, "/iesl/canvas/smurty/epiKB/MIL_data/entity_%s_type_dict.joblib" %(type_system))
