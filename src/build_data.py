'''

    Build data: Process CrossWikis, create pretrained embedding file, and vocab mapping
    TODO : perform redirects (!!)

'''
import bz2, os
import gzip
from collections import defaultdict as ddict
import joblib
import time
import shelve
import unicodedata
import numpy as np
import re
import copy
from tqdm import tqdm
from config import Config
import heapq
import numpy as np


PAD=0
OOV=1
vec_file_size = 2196017
crosswikis_size = 297073139







# ===== Code for query expansion and creating the mention set that we use to filter cross-wikis goes here ====


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


def getLnrm(arg, pattern):
  """Normalizes the given arg by stripping it of diacritics, lowercasing, and
  removing all non-alphanumeric characters.
  """
  arg = pattern.sub('', arg)
  arg = arg.lower()

  return arg


# ===================== Build Pretrained Vectors and vocab =====================

def dfs(node, ancestor, adj_matrix, transitive_closure):
    transitive_closure[node][ancestor] = 1.0

    for _parent in xrange(len(adj_matrix[ancestor])):
        if adj_matrix[ancestor][_parent]:
            dfs(node, _parent, adj_matrix, transitive_closure)

    return


def run_transitive_closure(adj_matrix):
    num_nodes = len(adj_matrix)
    transitive_closure = copy.deepcopy(adj_matrix)
    for node in xrange(num_nodes):
        dfs(node, node, adj_matrix, transitive_closure)

    return transitive_closure


def build_maps_figer(entity_file_path, type_file):
    type_dict = {}
    f = open(type_file)

    for line in f:
        _type = line.strip()
        if _type not in type_dict:
            type_dict[_type] = len(type_dict)


    f = open(entity_file_path)
    entity_dict = {"PAD" : 0}
    entity_type_map = {}



    for line in f:
        ent, _, types = line.strip().split("\t")
        all_types = types.split(",")

        entity_dict[ent] = len(entity_dict)
        entity_type_map[ent] = all_types
        for _type in all_types:
            if _type not in type_dict:
                type_dict[_type] = len(type_dict)

    for ent in entity_type_map:
        all_types_true = map(lambda _type : type_dict[_type], entity_type_map[ent])
        entity_type_map[ent] = all_types_true


    return entity_dict, type_dict, entity_type_map

def build_maps(entity_file_path, type_file_path):
    type_dict = {"NO_TYPES" : 0}

    f = open(type_file_path)
    all_edges = set()

    for line in f:
        node, parent = line.strip().split("->")
        node = node.strip()
        parent = parent.lstrip()

        if node not in type_dict:
            type_dict[node] = len(type_dict)
        if parent not in type_dict:
            type_dict[parent] = len(type_dict)

        all_edges.add((type_dict[node], type_dict[parent]))


    adj_matrix = np.zeros((len(type_dict), len(type_dict)))

    for node, parent in all_edges:
        adj_matrix[node][parent] = 1.0

    transitive_closure = run_transitive_closure(adj_matrix)

    for node in xrange(len(transitive_closure)):
        transitive_closure[node][node] = 0.0

    f.close()

    print("Found %d types" %len(type_dict))
    f = open(entity_file_path)
    entity_dict = {"PAD" : 0}
    entity_type_map = {}


    for line in f:
        ent, _, types = line.strip().split("\t")
        all_types = types.split(",")

        entity_dict[ent] = len(entity_dict)
        entity_type_map[ent] = all_types


    for ent in entity_type_map:
        all_types_true = map(lambda _type : type_dict[_type], entity_type_map[ent])
        entity_type_map[ent] = all_types_true

    return type_dict, entity_dict, entity_type_map, transitive_closure


def build_types(file_path):
    f = open(file_path)
    type_dict = {"NO_TYPES": 0}
    idx = 1
    for line in f:
        curr_type = line.strip()
        if (curr_type not in type_dict):
            type_dict[curr_type] = idx
            idx += 1

    print("-"*50)
    print("Found %d types" %(len(type_dict)))
    return type_dict

def build_entities(file_path):
    f = open(file_path)

    entity_dict = {"PAD" : 0}
    idx = 1
    for line in f:
        curr_ent = line.strip().split('\t')[0]
        if (curr_ent not in entity_dict):
            entity_dict[curr_ent] = idx
            idx += 1


    print("-"*50)
    print("Found %d entities." %(len(entity_dict)))

    return entity_dict


def build_vocab_and_embeddings(glove_file, trimmed_file, dim):
    vocab = {"PAD" : 0, "UNK" : 1, "<target>" : 2, "</target>" : 3}
    idx = 4
    trained_dict ={}
    with open(glove_file) as f:
        for line in tqdm(f, total = vec_file_size):
            line = line.rstrip().split()
            if (len(line) != dim + 1):  # hardcode
                continue
            word = line[0].strip()
            if word not in vocab:
                vocab[word] = idx
                idx += 1

            embedding = list(map(float, line[1:]))
            trained_dict[word] = np.array(embedding).astype(np.float32)


    trained_embeddings = np.random.randn(len(vocab), dim)

    for word in vocab:
        idx = vocab[word]
        if word in trained_dict:
            trained_embeddings[idx] = trained_dict[word]

    print("Vocab size: %d" %(len(vocab)))
    np.savez_compressed(trimmed_file,embeddings=trained_embeddings)
    return vocab



def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    return np.load(filename)["embeddings"]


# =============== build typing data ==============

def get_typing_data():
    f =open("/iesl/canvas/smurty/epiKB/AIDA_linking/AIDA_original/AIDA_entities.txt", "r"); lines = f.readlines(); f.close()
    wiki2fb = {}
    for line in lines:
        ent, fb = line.strip().split("\t")
        all_ent = fb.split(" ")
        wiki2fb[ent] = all_ent

    fb2types = {}
    f = gzip.open("/iesl/canvas/pat/data/freebase/entity_to_fbtypes.gz"); lines = f.readlines(); f.close()

    for line in lines:
        if len(line.strip().split("\t")) != 2:
            print(line)
            continue
        ent, types = line.strip().split("\t")
        types = set(types.split(","))
        fb2types[ent] = types

    fbtypes2figer = {}
    all_figers = set()
    f = open("/iesl/canvas/pat/data/freebase/figer_type.map"); lines = f.readlines(); f.close()
    for line in lines:
        fbtype, figer_type = line.strip().split("\t")
        fbtypes2figer[fbtype] = figer_type
        all_figers.add(figer_type)

    def get_figer(_type):
        if _type not in fbtypes2figer:
            dat = _type[1:].split("/")
            if len(dat) != 2:
                return ""
            domain, sub_type = _type[1:].split("/")
            #domain = "/%s" %domain
            check = "/%s/%s" %(domain, domain)
            if check in fbtypes2figer:
                return fbtypes2figer[check]
            elif "/%s" %domain in all_figers:
                return "/%s" %domain
            else:
                return ""
        else:
            return fbtypes2figer[_type]

    f = open("/iesl/canvas/smurty/epiKB/AIDA_linking/AIDA_original/AIDA_entities_types.txt", "w")
    for ent in wiki2fb:
        fb_ents = wiki2fb[ent]
        all_types = set()
        for fb_ent in fb_ents:
            if fb_ent in fb2types:
                types_curr = set(map(lambda _type: get_figer(_type), fb2types[fb_ent]))
                all_types |= types_curr

        all_types = filter(lambda _type: len(_type) != 0, all_types)
        fb_ent = ",".join(fb_ents)
        if len(all_types) == 0:
            f.write("%s\t%s\tNO_TYPES\n" %(ent, fb_ent))
        else:
            all_figer_types = ",".join(all_types)
            f.write("%s\t%s\t%s\n" %(ent, fb_ent, all_figer_types))

    f.close()



if __name__ == "__main__":
    run_dir = os.getcwd()
    config_obj = Config(run_dir, "AIDA", "entity_linker_order")
    #type_dict, entity_dict, entity_type_map, adj_matrix = \
    #        build_maps(config_obj.raw_entity_type_file, config_obj.raw_type_file)

    entity_dict, type_dict, entity_type_map =  build_maps_figer(config_obj.raw_entity_type_file, config_obj.raw_type_file)



    print("Number of types: %d" %len(type_dict))
    print("numer of entities: %d" %len(entity_dict))

    print("Dumping to files")
    joblib.dump(type_dict, config_obj.type_file)
    joblib.dump(entity_dict, config_obj.entity_file)
    joblib.dump(entity_type_map, config_obj.entity_type_file)

    #joblib.dump(adj_matrix, config_obj.typenet_matrix)
    #get_typing_data()
    #crosswikis = CrossWikis(config_obj.cross_wikis_shelve)
    #crosswikis.read_dictionary(config_obj.crosswikis_file, config_obj.raw_entity_file, config_obj.redirects_file)


#    crosswikis.open_shelve()
#    candidates = crosswikis.getCandidates("obama")
#    print(candidates)



     #build entity and vocab joblibs
    #entities = build_entities(config_obj.raw_entity_file)
    #vocab = build_vocab_and_embeddings(config_obj.embedding_downloaded_file, config_obj.embedding_file, config_obj.embedding_dim)

    #joblib.dump(entities, config_obj.entity_file)
    #joblib.dump(vocab, config_obj.vocab_file)
