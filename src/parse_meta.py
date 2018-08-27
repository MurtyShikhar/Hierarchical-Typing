import xml.etree.ElementTree
from nltk.tokenize import sent_tokenize
import re
import joblib
import os
import glob
import copy
from tqdm import tqdm
import sys
import gzip

reload(sys)
sys.setdefaultencoding("utf-8")

def get_annotations(annotation_node):
    text_dict = {}
    for i, node in enumerate(annotation_node):
        curr_attribs = node.attrib
        # == get start and end nodes of the annotation ==
        start_node   = curr_attribs['StartNode']
        end_node     = curr_attribs['EndNode']
        text_dict[(start_node, end_node)] = {}

        # == get features ==
        for feature in node:
            if len(feature) != 2:
                continue
            attrib_curr = feature[0].text
            val_curr    = feature[1].text
            text_dict[(start_node, end_node)][attrib_curr] = val_curr

    return text_dict


def get_text(text_node, text_dict):
    all_text  = []
    total_text_nodes = len(text_node)
    raw_text = []
    all_entity_mentions = []

    regex_pattern = re.compile("C\d\d\d\d\d\d\d")

    corrupted = []
    for i,node in enumerate(text_node):
        token = node.tail
        if token is None or i == len(text_node) - 1: break
        else:
            nxt_node = text_node[i+1]
            _key = (node.attrib['id'], nxt_node.attrib['id'])

            if int(_key[0]) < 35: continue

            if _key not in text_dict:
                text_dict[_key] = {}

            text_dict[_key]['text'] = token
            raw_text.append(token)

            # check if token has type
            if 'SemanticID' in text_dict[_key]:
                sem_id  = text_dict[_key]['SemanticID']
                if 'UMLS' not in text_dict[_key]:
                    umls_id = "<none>"
                    corrupted.append(_key)
                else:
                    umls_id = text_dict[_key]['UMLS'].split(" ")[0]

                    if not regex_pattern.match(umls_id):
                        umls_id = "<none>"


                    all_entity_mentions.append((token, umls_id))
                    all_text.append("<target>")
                    all_text.append("%s,%s" %(sem_id, umls_id))
                    all_text.append("%s" % "_".join(token.split(" ")))
                    all_text.append("</target>")
            else:
                all_text.append(token)

    return all_text, raw_text, corrupted, all_entity_mentions


def parse_file_helper(filename):
    try:
        root = xml.etree.ElementTree.parse(filename).getroot()
        if len(root) < 4:
            return None, None, None, None

        text_dict = get_annotations(root[3])

        all_text, raw_text, corrupted, all_entity_mentions = get_text(root[1], text_dict)

        #if len(corrupted):
        #    print("%s is corrupted" %filename, corrupted)

        return text_dict, " ".join(all_text), " ".join(raw_text), all_entity_mentions
    except:
        return None, None, None, None


def parse_file(filename):
    text_dict, all_text, raw_text, all_entity_mentions = parse_file_helper(filename)
    if text_dict is None:
        return None, None
    sent_tokenize_list = [re.sub( '\s+', ' ', sent).strip() for sent in sent_tokenize(all_text)]

    f = open("meta_data_parsed/%s" %filename.split("/")[-1], "w")
    for sent in sent_tokenize_list:
        f.write("%s\n" %sent)

    f.close()
    return sent_tokenize_list, all_entity_mentions


def parse_meta(filenames):
    all_data = []
    all_entity_mentions = []
    not_annotated = 0

    for filename in filenames:
        print("processing: %s" %filename)
        curr, ent_mentions = parse_file(filename)
        if curr is None: not_annotated +=1
        else:
            all_data += curr[:-1]
            all_entity_mentions += ent_mentions

    print("%d files not annotated" %not_annotated)

    return all_data


# ===== Functions for processing data further
def get_jth_mention(j, sentence):
    i = 0
    mention_num = 0
    curr_sentence = []
    gold_tag     = None
    gold_ent     = None
    gold_mention = None
    while i < len(sentence):
        tok = sentence[i]

        if tok == "<target>":
            mention_num += 1
            i += 1
            tag, ent = sentence[i].split(",")
            i += 1
            if mention_num == j:
                curr_sentence.append("<target>")
                gold_ent = ent
                gold_tag = tag

            curr_mention = sentence[i]
            i += 1

            if mention_num == j:
                gold_mention = " ".join(curr_mention.split("_"))
                curr_sentence.append(curr_mention)
                curr_sentence.append("</target>")
            else:
                curr_sentence.append(" ".join(curr_mention.split("_")))


        elif tok != "</target>":
            curr_sentence.append(sentence[i])
        i += 1


    if curr_sentence.count("<target>") != 1:
        print("orig:", sentence)
        print("new:", curr_sentence)
    if curr_sentence.count("</target>") != 1:
        print("orig:", sentence)
        print("new:", curr_sentence)

    return " ".join(curr_sentence), gold_tag, gold_ent, gold_mention



def convert(sentence, deleted_entities):
    '''
        Takes a sentence with target annotations and returns a list of sentences where only one mention is a target sentence
    '''

    sentence = sentence.split(" ")
    sent_list = []

    # find the number of mentions in the sentence

    num_mentions = 0
    for tok in sentence:
        if tok == "<target>":
            num_mentions += 1

    tags = set()
    entities = set()

    for j in xrange(num_mentions):
        curr_sentence, gold_tag, gold_ent, gold_mention = get_jth_mention(j+1, sentence)

        if gold_ent in deleted_entities:
            continue

        sent_list.append("%s\t%s\t%s\t%s" %(gold_ent, gold_tag, gold_mention, curr_sentence))
        entities.add(gold_ent)
        tags.add(gold_tag)

    return sent_list, entities, tags


def convert_data(data, deleted_entities):

    entities = set()
    tags = set()

    processed_data = []

    for sent in data:
        curr_processed, curr_entities, curr_tags  = convert(sent, deleted_entities)
        processed_data += curr_processed

        entities |= curr_entities
        tags |= curr_tags

    return processed_data, entities, tags



# ======= Functions for the ALIAS TABLE =========
def read_alias_table(path):
    alias_table = {}

    for line in tqdm(gzip.open(path), total = 8375000):
        mention, entity, prob = line.strip().split("\t")
        if mention not in alias_table:
            alias_table[mention] = {}

        if entity in alias_table[mention]: continue
        alias_table[mention][entity] = float(prob)

    alias_table_new = {}
    for mention in alias_table:
        alias_table_new[mention] = alias_table[mention].items()

    return alias_table_new


# ===== Functions for creating the UMLS ontology

def dfs(node, root_node, linked_list, linked_list_closure, visited):
    if node != root_node:
        linked_list_closure[root_node].add(node)

    if node in visited:
        return

    visited.add(node)
    if node not in linked_list:
        return

    for parent_node in linked_list[node]:
        dfs(parent_node, root_node, linked_list, linked_list_closure, visited)

def run_transitive_closure(linked_list):
    transitive_closure = copy.deepcopy(linked_list)
    print("computing transitive closure.")
    for node in tqdm(linked_list, total = len(linked_list)):
        visited = set()
        dfs(node, node, linked_list, transitive_closure, visited)

    return transitive_closure


def create_hierarchy(path, entity_dict):
    entity_hierarchy_linked_list = {}

    for line in gzip.open(path):
        child, parent = line.strip().split("\t")
        if child not in entity_dict or parent not in entity_dict:
            continue
        child = entity_dict[child]
        parent = entity_dict[parent]
        if child not in entity_hierarchy_linked_list:
            entity_hierarchy_linked_list[child] = set()

        entity_hierarchy_linked_list[child].add(parent)

    return run_transitive_closure(entity_hierarchy_linked_list), entity_hierarchy_linked_list


if __name__ == '__main__':
    '''
    filenames = glob.glob("meta_data/*")
    if not os.path.exists("meta_data_parsed"):
        os.makedirs("meta_data_parsed")

    len_f = len(filenames)
    train = filenames[: int(0.8*len_f)]
    dev   = filenames[int(0.8*len_f) : int(0.9*len_f)]
    test  = filenames[int(0.9*len_f) : ]

    deleted_entities = set([line.strip() for line in open("/iesl/data/umls_2017AB/deleted_concepts.txt") ])


    train, entity_train, tag_train = convert_data(parse_meta(train), deleted_entities)
    dev, entity_dev, tag_dev = convert_data(parse_meta(dev), deleted_entities)
    test, entity_test, tag_test = convert_data(parse_meta(test), deleted_entities)

    entity_set = entity_train | entity_dev | entity_test
    tag_set    = tag_train | tag_dev | tag_test

    entity_dict = {ent : idx for (idx, ent) in enumerate(entity_set)}
    tag_dict    = {tag : idx for (idx, tag) in enumerate(tag_set)}


    # ===== Dump train/dev/test and dictionaries ===
    print("Train: %d, Dev: %d, Test: %d" %(len(train), len(dev), len(test)))

    joblib.dump(train, "meta_data_processed/meta_train.joblib")
    joblib.dump(dev  , "meta_data_processed/meta_dev.joblib")
    joblib.dump(test , "meta_data_processed/meta_test.joblib")

    joblib.dump(entity_dict, "meta_data_processed/entities.joblib")
    joblib.dump(tag_dict, "meta_data_processed/types.joblib")

    # == get UMLS ontology ==
    entity_hierarchy_linked_list, entity_hierarchy_linked_list_orig = create_hierarchy("/iesl/data/umls_2017AB/parent_broader_relations.gz", entity_dict)
    joblib.dump(entity_hierarchy_linked_list, "meta_data_processed/entity_hierarchy.joblib")
    joblib.dump(entity_hierarchy_linked_list_orig, "meta_data_processed/entity_hierarchy_orig.joblib")
    '''
    # == get alias table ==
    alias_table = read_alias_table("/iesl/canvas/pat/epiKB/data/umls_candidate_file.gz")
    joblib.dump(alias_table, "meta_data_processed/crosswikis.shelve")
