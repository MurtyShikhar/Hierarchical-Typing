import numpy as np
import os
import glob
import copy
import re
from collections import defaultdict as ddict
from build_data import PAD, OOV, getLnrm
import joblib
import gzip
from config import Config
import unicodedata
import sys


reload(sys)
sys.setdefaultencoding("utf-8")


skip_tokens = set(["<em>", "</em>"])
target_mention_markers = set(["<target>", "</target>"])
AIDA_ATTRIBS = ['mention_representation']
WIKI_ATTRIBS = ['left_contexts', 'right_contexts', 'entity_candidates', 'gold_ids', 'priors']
MIL_ATTRIBS  = ['mention_representation', 'context', 'position_embeddings', 'gold_types', 'ent', 'st_ids', 'en_ids']
MAX_SENT = 200
pattern = re.compile('[^\s\w]+')
'''
    Now, it directly returns minibatches over the training data by first loading in a file, and then returning a batch of examples directly

'''

class Batch():
    def __init__(self, data_attributes):
        self.data = ddict(list)
        self.attributes = data_attributes
        self._len = 0

    def __len__(self):
        return self._len

    def add_items(self, items):
        for attrib in self.attributes:
            self.data[attrib].append(items[attrib])
        self._len += 1


    def clear(self):
        for attrib in self.attributes:
            self.data[attrib] = []
        self._len = 0


# ======== Custom Dataset Iterators

class MILDataset():
    def __init__(self, entity_bags_dict, entity_type_dict, batch_attributes, entities, vocab_dict, embeddings, encoder, batch_size, num_fb_types, transformer, bag_size, entity_dict, cross_wikis, train = True):

        '''
            :entity_bags_dict: a dictionary containing a list of mentions for each entity
            :entity_types: a dictionary containing types for every entity (only contains freebase types), entity_types[ent] is a bit-vector
            :batch_attributes: the items contained in every data point in the batch
            :entities: the bags will be sampled only from these entities
            :vocab_dict: an index mapping words to ids
            :embeddings: pretrained word embeddings
            :batch_size and bag_size: number of items in the batch, number of items in the bag
        '''

        # batch size must be divisble by bag size
        self.entity_bags_dict = entity_bags_dict
        self.entity_type_dict = entity_type_dict


        self.batch_attributes = copy.deepcopy(batch_attributes)
        self.entities = entities


        self.vocab_dict = vocab_dict

        # == needed for entity linking
        self.entity_dict = entity_dict
        self.cross_wikis = cross_wikis


        self.embeddings = embeddings
        self.bag_size = bag_size
        self.batch_size = batch_size

        self.transformer = transformer
        self.num_fb_types = num_fb_types

        self.train = train
        self.encoder_type = encoder

    def get_batch(self):
        minibatch = Batch(self.batch_attributes)
        for ent in self.entities:
            if len(minibatch) == self.batch_size*self.bag_size:
                yield minibatch
                minibatch.clear()

            all_mentions = self.entity_bags_dict[ent]
            assert(len(all_mentions) >= self.bag_size)

            list_items = self.transformer(all_mentions, self.vocab_dict, self.entity_dict, self.cross_wikis, self.embeddings, self.bag_size, self.train, self.encoder_type)
            bit_vec = [0]*self.num_fb_types  # predictions are made only for freebase types
            for _type in self.entity_type_dict[ent]:
                assert(_type < self.num_fb_types)
                bit_vec[_type] = 1.0


            for items in list_items:
                items['gold_types'] = bit_vec
                items['ent'] = ent
                minibatch.add_items(items)

        if len(minibatch) != 0:
            yield(minibatch)


class LinkingDataset(object):
    def __init__(self, data, batch_attributes, entity_dict, vocab_dict, type_dict, cross_wikis, embeddings, batch_size, line_transformer, train, encoder_type):
        self.data = data
        self.batch_attributes = copy.deepcopy(batch_attributes)

        self.embeddings = embeddings
        self.cross_wikis = cross_wikis

        self.entity_dict = entity_dict
        self.vocab_dict  = vocab_dict
        self.type_dict   = type_dict

        self.batch_size = batch_size
        self.line_transformer = line_transformer
        self.train = train
        self.encoder_type = encoder_type



    def get_batch(self):
        minibatch = Batch(self.batch_attributes)
        for line in self.data:
            if len(minibatch) == self.batch_size:
                yield minibatch
                minibatch.clear()

            items, add_to_minibatch = self.line_transformer(line, self.entity_dict, self.vocab_dict, self.type_dict, self.embeddings, self.cross_wikis, self.train, self.encoder_type)
            # this variable takes care of not adding noisy/garbage sentences to the minibatch. Doesn't apply while evaluating.
            if add_to_minibatch:
                minibatch.add_items(items)

        if len(minibatch) != 0:
            yield minibatch




class SegmenterDataset(object):
    def __init__(self, data, batch_attributes, vocab_dict, char_dict, batch_size, line_transformer, train):
        self.vocab_dict = vocab_dict
        self.char_dict = char_dict

        self.data = data
        self.batch_size = batch_size
        self.batch_attributes = batch_attributes
        self.train = train
        self.line_transformer = line_transformer


    def get_batch(self):
        minibatch = Batch(self.batch_attributes)
        for line in self.data:
            if len(minibatch) == self.batch_size:
                yield minibatch
                minibatch.clear()

            items = self.line_transformer(line, self.vocab_dict, self.char_dict, self.train)

            minibatch.add_items(items)

        if len(minibatch) != 0:
            yield minibatch



# ===== Helper functions for datasets

def process(vocab_dict, tokens, flag_wiki = False, encoder = "rnn_phrase"):
    '''
    Takes the tokens of a mention sentence in the form of a list of strings, and outputs a left context and a right context list along with the mention surface form
    '''

    def process_tokens(tokens):
        p_tokens = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "<target>":
                p_tokens.append(tok)
                p_tokens += tokens[i+1].split("_")
                i += 1
            elif tok not in skip_tokens:
                p_tokens.append(tok)
            i += 1
        return p_tokens

    def find_str(text, str):
        for i, text in enumerate(text):
            if (text == str):
                return i
        return -1

    def getid(token):
        if token not in vocab_dict:
            return OOV
        else:
            return vocab_dict[token]


    if flag_wiki:
        tokens = process_tokens(tokens)

    i = find_str(tokens, "<target>")  # i+1..j-1 is the mention
    j = find_str(tokens, "</target>")
    assert(i != -1)
    assert(j != -1)
    # surface form of the mention is everything from i+1..j-1
    sfm_mention = tokens[i+1:j]


    left_context = tokens[: i+1] + sfm_mention
    right_context = sfm_mention + tokens[j:]

    left_context =  [getid(tok) for tok in left_context]  # s1... <target> m1 ... mN
    right_context = [getid(tok) for tok in right_context] # m1 ... mN </target> s_k+1 ... sN


    if encoder == "position_cnn" or encoder == "rnn_phrase":
        position_left = [MAX_SENT-dist for dist in xrange(i, 0, -1)]
        position_right = [MAX_SENT+dist - j for dist in xrange(j+1, len(tokens))]

        position_mention = [MAX_SENT]*(j-i -1)


        return [getid(tok) for tok in tokens if tok not in target_mention_markers], \
            [getid(tok) for tok in sfm_mention], position_left + position_mention + position_right, [i], [j-2]

    else:
        return [getid(tok) for tok in tokens], \
            [getid(tok) for tok in sfm_mention], [getid(tok) for tok in tokens], -1, -1


def getLnrm(arg, pattern):
  """Normalizes the given arg by stripping it of diacritics, lowercasing, and
  removing all non-alphanumeric characters.
  """
  arg = pattern.sub('', arg)
  arg = arg.lower()

  return arg


# At test time, we do NOT add the gold entity to the candidate set.
def get_candidates(sfm_mention, entity_dict, gold_entity, crosswikis, train):
    '''
        Use cross-wikis to retrieve top-100 entities for this mention
    '''
    candidate_probab_list = crosswikis[sfm_mention] if sfm_mention in crosswikis else []
    # the list is organized as (ent, prob), and we sort it in decreasing order of P(ent | sfm_mention)
    candidate_probab_list.sort(key = lambda item : -item[1])

    gold_id = entity_dict[gold_entity]

    if train:
        # take the top 99 and add to it the gold entity at train time
        crosswiki_data = [ (entity_dict[ent], prob) for (ent, prob) in candidate_probab_list if ent != gold_entity and ent in entity_dict][:199]

        gold_prob = 0.0
        for ent, prob in candidate_probab_list:
            if ent == gold_entity:
                gold_prob = prob
                break

        crosswiki_data.append((gold_id, gold_prob))
        candidates, priors = zip(*crosswiki_data)

        return candidates, priors, gold_id

    else:
        # take the top 100 entities and hope that the gold is somewhere in this
        crosswiki_data = [ [entity_dict[ent], prob] for (ent, prob) in candidate_probab_list if ent in entity_dict][:200]
        if (len(crosswiki_data) != 0):
            candidates, priors = zip(*crosswiki_data)
        else:
            candidates, priors = [], []

        return candidates, priors, gold_id


def process_type_data(vocab_dict, tokens, ent_st_id, ent_en_id):

    def getid(token):
        if token not in vocab_dict:
            return OOV
        else:
            return vocab_dict[token]

    tokens = [tok for tok in tokens if tok not in skip_tokens]
    sfm_mention = tokens[ent_st_id: ent_en_id]

    left_context = tokens[: ent_st_id] +sfm_mention
    right_context = sfm_mention + tokens[ent_en_id:]

    left_context =  [getid(tok) for tok in left_context] # s1... <target> m1 ... mN
    right_context = [getid(tok) for tok in right_context] # m1 ... mN </target> s_k+1 ... sN


    return left_context, right_context[::-1], [getid(tok) for tok in tokens], [getid(tok) for tok in sfm_mention]



# ========== A bunch of line transformers for parsing lines in different datasets. Currently only AIDA and wiki works with the new batching interface =====

def UMLS_transformer(mention_sentence, entity_dict, vocab_dict, type_dict, embeddings, cross_wikis, train, encoder_type):
    data = {}
    gold_ent, gold_tag, gold_mention, curr_sentence = mention_sentence.split("\t")
    sentence, sfm_mention, position_embedding, st_id, en_id = process(vocab_dict, curr_sentence.split(" "), flag_wiki=True, encoder = encoder_type)

    mention_representation = np.array([embeddings[_id] for _id in sfm_mention]).mean(axis = 0)
    data['mention_representation'] = mention_representation
    data['context'] = sentence
    data['position_embeddings'] = position_embedding
    data['st_ids'] = st_id
    data['en_ids'] = en_id
    #data['gold_tag'] = type_dict[gold_tag]


    all_candidates, priors, gold_id = get_candidates(gold_mention, entity_dict, gold_ent, cross_wikis, train)
    data['entity_candidates'] = all_candidates
    data['gold_ids'] = gold_id
    data['priors'] = priors

    return data, True


# 0 -> B, 1 -> I, 2 -> O
def get_boundaries(bio):
    """
    Extracts an ordered list of boundaries. BIO label sequences are
    -     Raw BIO: B     I     I     O => {(0, 2)}
    """
    boundaries= set()
    boundary_points = []
    i = 0

    while i < len(bio):
        if bio[i] == 2: i += 1
        else:
            s = i
            i += 1
            while i < len(bio) and bio[i] == 1:
                i += 1
            boundaries.add((s, i - 1))
            boundary_points.append(s)
            boundary_points.append(i-1)

    return boundaries, boundary_points


# 0 -> B, 1 -> I, 2 -> O
def SegmenterTransformerAIDA(line, vocab_dict, char_dict, train):
    def getid(tok):
        if tok in vocab_dict:
            return vocab_dict[tok]
        else:
            return OOV

    def getcharid(tok):
        if tok in char_dict:
            return char_dict[tok]
        else:
            return OOV

    def process_sentence(tokens):
        gold_ids = []
        p_tokens = []

        i = 0
        prev_tag = None
        while i < len(tokens):
            tok, _, _, tag = tokens[i]
            p_tokens.append(tok)
            if tag == 'O':
                gold_ids.append(2)
            elif tag == prev_tag:
                gold_ids.append(1)
            else:
                gold_ids.append(0)

            prev_tag = tag
            i += 1

        return gold_ids, p_tokens

    gold_ids, p_tokens = process_sentence(line)

    _, pointers = get_boundaries(gold_ids)
    char_tokens_fwd = [ [getcharid(char) for char in word] for word in p_tokens ]

    if not (all(len(tok) != 0 for tok in p_tokens)):
        print(curr_sentence)
        print(p_tokens)
        sys.exit(-1)

    assert(len(gold_ids) == len(p_tokens))
    p_tokens = [getid(tok) for tok in p_tokens]

    data = {'context' : p_tokens, 'gold_ids' : gold_ids, 'char_ids' : char_tokens_fwd, 'gold_ids_pointer' : pointers}

    return data


# 0 -> B, 1 -> I, 2 -> O
def SegmenterTransformer(line, vocab_dict, char_dict, train):
    curr_sentence = filter(lambda word: len(word) != 0, get_curr_sentence(line))

    def getid(tok):
        if tok in vocab_dict:
            return vocab_dict[tok]
        else:
            return OOV

    def getcharid(tok):
        if tok in char_dict:
            return char_dict[tok]
        else:
            return  OOV


    def process_sentence(tokens):
        gold_ids = []
        p_tokens = []

        def process_target(i):
            i += 1

            mention = tokens[i].split("_")
            # first one gets a B tag
            first = True
            for m in mention:
                if (len(m) != 0):
                    p_tokens.append(m)
                    gold_ids.append(0 if first else 1)
                    first = False

            # i+1 = </target>, i+2 = </em>
            i += 3
            return i


        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "<em>":
                i += 1
                if tokens[i] == "<target>":
                    i = process_target(i)
                else:
                    p_tokens.append(tokens[i])
                    gold_ids.append(0)

                    i += 1
                    while tokens[i] != "</em>":
                        p_tokens.append(tokens[i])
                        gold_ids.append(1)
                        i += 1

                    # move past </em>
                    i += 1

            else:
                p_tokens.append(tok)
                gold_ids.append(2)

                i += 1

        return gold_ids, p_tokens

    gold_ids, p_tokens = process_sentence(curr_sentence)

    _, pointers = get_boundaries(gold_ids)
    char_tokens_fwd = [ [getcharid(char) for char in word] for word in p_tokens ]
    if not (all(len(tok) != 0 for tok in p_tokens)):
        print(curr_sentence)
        print(p_tokens)
        sys.exit(-1)

    assert(len(gold_ids) == len(p_tokens))
    p_tokens = [getid(tok) for tok in p_tokens]

    data = {'context' : p_tokens, 'gold_ids' : gold_ids, 'char_ids' : char_tokens_fwd, 'gold_ids_pointer' : pointers}

    return data


def get_curr_sentence(mention_raw):
    mention_raw = mention_raw.split("\t")
    assert(len(mention_raw) == 6)
    title, sfm_mention, entity, prev_sentence, curr_sentence, next_sentence = mention_raw

    return curr_sentence.split(" "), sfm_mention, entity


def MILtransformer(all_mentions, vocab_dict, entity_dict, cross_wikis, embeddings, bag_size, train, encoder_type):
    '''
        Transformer for MIL data.
        all_mentions: a bag for mentions for the current entity
        vocab_dict: an index from word to its id
        embeddings: a set of pretrained embeddings
        bag_size: number of mentions we want to sample from all_mentions

        return: a list of dictionaries
    '''

    if train:
        subsampled_ids = np.random.choice(len(all_mentions), bag_size, replace=False)
    else:
        subsampled_ids = xrange(bag_size)

    subsampled_bag = [all_mentions[idx] for idx in subsampled_ids]

    all_data = []
    for curr_data in subsampled_bag:
        data = {}
        curr_sentence, gold_mention, gold_ent = get_curr_sentence(curr_data)
        gold_ent     = gold_ent[7:-4]
        gold_mention = " ".join(gold_mention[9:].split("_"))
        sentence, sfm_mention, position_embedding, st_id, en_id = process(vocab_dict, curr_sentence, flag_wiki=True, encoder = encoder_type)
        
        all_candidates, priors, gold_id = get_candidates(getLnrm(gold_mention, pattern), entity_dict, gold_ent, cross_wikis, train)

        mention_representation = np.array([embeddings[_id] for _id in sfm_mention]).mean(axis = 0)
        data['mention_representation'] = mention_representation
        data['context'] = sentence
        data['position_embeddings'] = position_embedding
        data['st_ids'] = st_id
        data['en_ids'] = en_id
        data['entity_candidates'] = all_candidates
        data['priors'] = priors
        data['gold_ids'] = gold_id
        assert(len(position_embedding) == len(sentence))
        all_data.append(data)


    return all_data



def transform_sentence_wiki_typing(mention_raw, entity_dict, vocab_dict, embeddings, train, type_dict, num_types, encoder):

    data = {}

    entity_st, entity_en, curr_sentence, gold_types, _, = mention_raw

    entity_st = int(entity_st)
    entity_en = int(entity_en)

    curr_sentence = curr_sentence.split(" ")
    curr_sentence.insert(entity_st, "<target>")
    curr_sentence.insert(entity_en+1, "</target>")
    gold_types = gold_types.split(" ")

    if type_dict is not None:
        bit_vec = [0]*num_types

        for _type in gold_types:
            if _type not in type_dict:
                _type = "NO_TYPES"
            bit_vec[type_dict[_type]] = 1


    sentence, sfm_mention, position_embedding, st_id, en_id = process(vocab_dict, curr_sentence, flag_wiki=False, encoder = encoder)
    mention_representation = np.array([embeddings[_id] for _id in sfm_mention]).mean(axis = 0)
    data['ent'] = "<NONE>"
    data['mention_representation'] = mention_representation
    data['context'] = sentence
    data['position_embeddings'] = position_embedding
    data["gold_types"] = bit_vec

    data['st_ids'] = st_id
    data['en_ids'] = en_id
    assert(len(position_embedding) == len(sentence))


    return data, True



def transform_sentence_aida(mention_raw, entity_dict, vocab_dict, embeddings, train, type_dict, num_types):
    '''
    :param mention_raw
    :return a dictionary containing the attributes and the their values
    '''

    data = {}
    if len(mention_raw) != 2:
        print(mention_raw)
    entity, curr_sentence = mention_raw

    curr_sentence = curr_sentence.split(" ")

    if len(curr_sentence) <= 4 and train:
        return [[] for attrib in AIDA_ATTRIBS], False



    if type_dict is not None:
        _types = type_dict[entity]
        bit_vec = [0]*num_types
        for _type in _types:
            bit_vec[_type] = 1
        # list of all types that are true for the entity linked to this mention
        data["gold_types"] = bit_vec




    left_context, right_context, sentence, sfm_mention = process(vocab_dict, curr_sentence)

    mention_representation = np.array([embeddings[_id] for _id in sfm_mention]).mean(axis = 0)

    data['left_contexts'] = left_context
    data['right_contexts'] = right_context
    data['context'] = sentence

    data['mention_representation'] = mention_representation

    return data, True

def transform_sentence_wiki(mention_raw, entity_dict, vocab_dict, embeddings, train, type_dict, num_types):
    '''
    :param mention_raw:
    :return left_context, right_context, entity_candidates
    '''

    data = {}
    if (len(mention_raw) == 6):
        title, sfm_mention, entity, prev_sentence, curr_sentence, next_sentence = mention_raw
    elif len(mention_raw) == 5:
        title, sfm_mention, entity, s1, s2 = mention_raw
        if "<target>" not in s1:
            curr_sentence = s2
            prev_sentence = s1
            next_sentence = ""
        else:
            curr_sentence = s1
            next_sentence = s2
            prev_sentence = ""
    elif len(mention_raw) == 4:
        title, sfm_mention, entity, curr_sentence = mention_raw
        next_sentence = ""
        prev_sentence = ""
    else:
        return [[] for attrib in WIKI_ATTRIBS], False



    curr_sentence = curr_sentence.split(" ")
    prev_sentence = prev_sentence.split(" ")
    next_sentence = next_sentence.split(" ")


    # add extra context if the current sentence is too small

    # currently, get rid of sentence if it's too short or too long
    if len(curr_sentence) <= 10 or len(curr_sentence) >= 100:
        return [[] for attrib in WIKI_ATTRIBS], False


    if type_dict is not None:
        _types = type_dict[entity]
        # list of all types that are true for the entity linked to this mention

        bit_vec = [0]*num_types
        for _type in _types:
            bit_vec[_type] = 1
        data["gold_types"] =bit_vec

    '''
        Sentence = s1 ... sk, <target>, m1 ... mN, </target> ,s_k+1, ... sN
        Left context = s1 ... sk, <target>, m1 ... mN
        Right context = sN ... s_k+1, </target>, mN ... m1
        entity_candidates = <gold_entity>, e2, e3, ... , e30
    '''


    left_context, right_context, sentence, sfm_mention = process(vocab_dict, curr_sentence, True)
    #all_candidates, priors, gold_id = get_candidates(sfm_mention, entity_dict, entity, cross_wikis, train)

    mention_representation = np.array([embeddings[_id] for _id in sfm_mention]).mean(axis = 0)
    data['left_contexts'] = left_context
    data['right_contexts'] = right_context
    data['mention_representation'] = mention_representation
    data['context'] = sentence

    #data['entity_candidates'] = all_candidates
    #data['gold_ids'] = gold_id
    #data['priors'] = priors
    return data, True





def get_batch_coverage(entity_candidates, gold_ids):
    '''
    A diagnostic function to check how many times the gold entity is in the entity set
    '''
    coverage=0.0
    for curr_candidates, gold_id in zip(entity_candidates, gold_ids):
        if gold_id in curr_candidates:
            coverage += 1

    return coverage





if __name__ == '__main__':

    run_dir = os.getcwd()
    config_obj = Config(run_dir, "AIDA", "bl")


    train_files = glob.glob("%s/*.gz" %config_obj.train_file)
    dev_files = glob.glob("%s/*.gz" %config_obj.dev_file)
    test_files = glob.glob("%s/*.gz" %config_obj.test_file)

    entity_dict = joblib.load(config_obj.entity_file)
    type_dict = joblib.load(config_obj.type_file)
    entity_type_dict = joblib.load(config_obj.entity_type_file)
    typenet_matrix = joblib.load(config_obj.typenet_matrix)

    vocab_dict = joblib.load(config_obj.vocab_file)

    config_obj.vocab_size = len(vocab_dict)
    config_obj.entity_size = len(entity_dict)
    config_obj.type_size = len(type_dict)
    crosswikis = CrossWikis(config_obj.cross_wikis_shelve)
    crosswikis.open_shelve()
    print("num entities: %d" %(len(entity_dict)))


    dev_data = LinkingDataset(dev_files, AIDA_ATTRIBS, entity_dict, vocab_dict, crosswikis, config_obj.batch_size, transform_sentence_aida, train = False)
    test_data = LinkingDataset(test_files, AIDA_ATTRIBS, entity_dict, vocab_dict, crosswikis_obj, config_obj.batch_size, transform_sentence_aida, train = False)

    coverage = 0.0
    _len = 0.0
    for minibatch in dev_data.get_batch():
        coverage += get_batch_coverage(minibatch.data['entity_candidates'], minibatch.data['gold_ids'])
        _len += len(minibatch)


    print("coverage: %5.4f" %(coverage/_len))

    crosswikis.close_shelve()
