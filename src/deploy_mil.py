'''
    Code for training/testing/evaluating/predicting joint linking + typing + structure.
    UNK always maps to 0
    Predictions are made only for freebase types, and hence, for ease we map all freebase types for ids first, and then map wordnet types
'''


import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import *
from data_iterator import *
from build_data import *
from general_utils import *
import os
import torch.optim as optim
from config_mil import Config_MIL


logging.basicConfig(stream = sys.stdout, level=logging.INFO)
logger = logging.getLogger("MIL Typing + Linking + structure")

# == The main training/testing/prediction functions
def get_tensors_from_minibatch(minibatch, type_indexer, type_indexer_struct, typenet_matrix, num_iter, config, bag_size, train = True):
    mention_representation = np.array(minibatch.data['mention_representation']) # (batch_size*bag_size, embedding_dim)
    batch_size = len(mention_representation)/bag_size

    # possible mention types
    type_idx = type_indexer.repeat(batch_size*bag_size, 1) #(batch_size*bag_size, num_fb_types)
    gold_types = torch.FloatTensor(minibatch.data['gold_types']).view(-1, bag_size, config.fb_type_size)[:, 0, :] #(batch_size, num_types)
    mention_representation = torch.FloatTensor(mention_representation) #(batch_size*bag_size, embedding_dim)

    all_tensors = {'type_candidates' : type_idx, 'gold_types' : gold_types, 'mention_representation' : mention_representation}

    if config.struct_weight > 0 and train:
        # ========= STRUCTURE LOSS TENSORS ===========
        # get current batch of children types to optimize
        num_types = config.type_size
        st = (num_iter*batch_size)%num_types
        en = (st + batch_size)%num_types

        if en <= st:
            type_curr_child = range(st, num_types) + range(0, en)
        else:
            type_curr_child = range(st, en)

        type_curr_child = np.array(type_curr_child)
        batch_size_types = len(type_curr_child)


        all_tensors['structure_parents_gold'] = torch.FloatTensor(typenet_matrix[type_curr_child].reshape(batch_size_types, num_types)) # get parent of children sampled
        all_tensors['structure_parent_candidates'] = type_indexer_struct.repeat(batch_size_types, 1)
        all_tensors['structure_children'] = torch.LongTensor(type_curr_child.reshape(batch_size_types, 1)) # convert children sampled to tensors


        # == add entity to type structure loss if we have linking weights

        if config.linker_weight > 0:
            num_entities = config.entity_size
            ent_st = (num_iter*batch_size) % num_entities
            ent_en = (ent_st + batch_size) % num_entities

            if ent_en <= ent_st:
                sampled_child_nodes = range(ent_st, num_entities) + range(0, ent_en)
            else:
                sampled_child_nodes = range(ent_st, ent_en)

            batch_size_struct   = len(sampled_child_nodes)
            parent_candidates   = []

            parent_candidate_labels = []
            for node in sampled_child_nodes:
                curr_labels = []

                _node = inverse_entity_dict[node] #entity_type_dict maps entity lexical form to a set of types

                parent_candidates_curr = np.random.choice(num_types, config.parent_sample_size, replace = False) # choose a random set of parent links
                for parent_candidate in parent_candidates_curr:
                    if _node not in entity_type_dict:
                        curr_labels.append(0)
                    elif parent_candidate in entity_type_dict[_node]:
                        curr_labels.append(1)
                    else:
                        curr_labels.append(0)

                parent_candidate_labels.append(curr_labels)
                parent_candidates.append(parent_candidates_curr)

            all_tensors['structure_parents_gold_entity'] = torch.FloatTensor(parent_candidate_labels) # (batch_size_struct, config.parent_sample_size)
            all_tensors['structure_parent_candidates_entity'] = torch.LongTensor(parent_candidates)   # (batch_size_struct, config.parent_sample_size)
            all_tensors['structure_children_entity']  = torch.LongTensor(sampled_child_nodes).view(batch_size_struct, 1) # (batch_size_struct, 1)



    if config.encoder == "rnn_phrase":
        all_tensors['st_ids'] = torch.LongTensor(minibatch.data['st_ids'])
        print("s1", all_tensors['st_ids'].shape)
        all_tensors['en_ids'] = torch.LongTensor(minibatch.data['en_ids'])

    all_tensor_names = all_tensors.keys()
    for tensor_name in all_tensors.keys():
        if config.gpu:
            all_tensors[tensor_name] = Variable(all_tensors[tensor_name].cuda())
        else:
            all_tensors[tensor_name] = Variable(all_tensors[tensor_name])


    if config.linker_weight > 0:
        # ========= LINKING LOSS tensors ============
        all_tensors['entity_candidates'], all_tensors['entity_candidate_lens'] = \
            pad_sequences(minibatch.data['entity_candidates'], PAD, torch.LongTensor, config.gpu) #(batch_size*bag_size, 100)

        all_tensors['priors'], _ = pad_sequences(minibatch.data['priors'], 0.0, torch.FloatTensor, config.gpu) #(batch_size*bag_size, 100)


    # == feed to mention typer and get scores ==
    sentences , _ = pad_sequences(minibatch.data['context'], PAD, torch.LongTensor, config.gpu)
    position_embeddings, _ = pad_sequences(minibatch.data['position_embeddings'], PAD, torch.LongTensor, config.gpu, pos = True)


    all_tensors['sentences'] = sentences #.view(batch_size, config.bag_size, -1)
    all_tensors['position_embeddings'] = position_embeddings  #.view(batch_size, config.bag_size, -1)

    return all_tensors

def train(model, train, dev, test, config, typenet_matrix):

    # define the criterion for computing loss : currently only supports cross entropy
    loss_criterion = nn.BCEWithLogitsLoss()
    loss_criterion_linking = nn.CrossEntropyLoss()


    # ===== Different L2 regularization on the structure weights =====

    base_parameters = []
    struct_parameters = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        print(name, param.is_leaf)
        if name.startswith('bilinear_matrix'):
            struct_parameters.append(param)
        else:
            base_parameters.append(param)

    if config.struct_weight > 0:
        optimizer = optim.Adam([{'params': struct_parameters, 'weight_decay' : config.bilinear_l2}, {'params': base_parameters}], lr=config.lr, eps=config.epsilon,
                           weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
    else:
        optimizer = optim.Adam(base_parameters, lr=config.lr, eps=config.epsilon,
                           weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
    


    # transfer to GPU
    if config.gpu:
        model.cuda()


    best_accuracy = 0.0 #evaluate(model, dev, config)
    logger.info("\n==== Initial accuracy: %5.4f\n" %best_accuracy)

    num_fb_types = config.fb_type_size
    num_types = config.type_size

    type_indexer = torch.LongTensor(np.arange(num_fb_types)).unsqueeze(0)  #(1, num_fb_types) for indexing into the freebase part of type indexing array
    type_indexer_struct = torch.LongTensor(np.arange(num_types)).unsqueeze(0) #(1, num_types) for indexing into type embeddings array

    num_iter = 0
    patience = 0
    for epoch in xrange(config.num_epochs):
        logger.info("\n=====Epoch Number: %d =====\n" %(epoch+1))
        for minibatch in train.get_batch():
            all_tensors = get_tensors_from_minibatch(minibatch, type_indexer, type_indexer_struct, typenet_matrix, num_iter, config, config.bag_size)

            if config.encoder == "rnn_phrase":
                aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
            else:
                aux_data = all_tensors['position_embeddings']

            # set back to train mode
            model.train()

            struct_data  = [all_tensors['structure_children'], all_tensors['structure_parent_candidates']] if config.struct_weight > 0 else None
            struct_data_entity  = [all_tensors['structure_children_entity'], all_tensors['structure_parent_candidates_entity']] if (config.struct_weight > 0 and config.linker_weight > 0) else None
            linking_data = [all_tensors['entity_candidates'], all_tensors['priors']] if config.linker_weight > 0 else None
            logit_dict   = model(all_tensors['type_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], aux_data, config.bag_size, struct_data, linking_data, struct_data_entity)

            typing_loss = loss_criterion(logit_dict['typing_logits'], all_tensors['gold_types'])
            structure_loss = loss_criterion(logit_dict['type_structure_logits'], all_tensors['structure_parents_gold']) if config.struct_weight > 0 else 0.0
            structure_loss_entities = loss_criterion(logit_dict['entity_structure_logits'], all_tensors['structure_parents_gold_entity']) if (config.struct_weight > 0 and config.linker_weight > 0) else 0.0
            linker_loss  = loss_criterion_linking(logit_dict['linking_logits'], all_tensors['entity_candidate_lens'] - 1) if config.linker_weight > 0 else 0.0

            train_loss = config.typing_weight*typing_loss + config.struct_weight*(structure_loss + structure_loss_entities) + config.linker_weight*linker_loss
            step(model, train_loss, optimizer, config.clip_val)

            num_iter += 1
            if num_iter % 10 == 0:
                logger.info("\n=====Loss: %5.4f Step: %d ======\n" %(train_loss.data[0], num_iter))

            if num_iter % 500 == 0:
                curr_accuracy = evaluate(model, dev, config)
                logger.info("\n==== Accuracy on dev after %d iterations:\t %5.4f\n" % (num_iter, curr_accuracy))
                if curr_accuracy > best_accuracy:
                    best_accuracy = curr_accuracy
                    test_accuracy = evaluate(model, test, config) #"results_typing/acc_test_%s.txt" %config.model_file)
                    logger.info("\n==== Accuracy on test after %d iterations:\t %5.4f\n" % (num_iter, test_accuracy))
                    if config.save_model:
                        save_model(model, config)
                    patience = 0
                else:
                    patience += 1
                    if patience > 10000:
                        break



    # load the best model
    load_model(model, config)

    accuracy_test = evaluate(model, test, config)

    logger.info("\n===== Training Finished! =====\n")
    logger.info("\n===== Accuracy on Test: %5.4f \n" %accuracy_test)


def predict_linking(scores, entity_candidates):

    # scores is a batch_size x num_candidates tensor of logits.

    # == find the argmax and use that to index into entity_candidates ===
    _, idx = scores.max(dim = -1)
    return entity_candidates.gather(1, idx.unsqueeze(1)).squeeze(1)



def predict_typing(scores, gold_ids, threshold=0):
    true_and_prediction = []
    for score,true_label in zip(scores, gold_ids):
        predicted_tag = []
        true_tag = []
        for label_id,label_score in enumerate(list(true_label)):
            if label_score > 0:
                true_tag.append(label_id)
        lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
        predicted_tag.append(lid)
        for label_id,label_score in enumerate(list(score)):
            if label_score >= threshold: #or label_score == score[lid]:
                if label_id != lid:
                    predicted_tag.append(label_id)
        true_and_prediction.append((true_tag, predicted_tag))

    return true_and_prediction

def evaluate(model, eval_data, config, log_file = None):
    accuracy = 0.0
    _len = 0
    coverage=0.0
    num_types = model.config.fb_type_size # only index into freebase partition
    _ap = 0.0
    test_bag_size = config.test_bag_size

    model.eval()
    if log_file is not None:
        f = open(log_file, "w")


    type_indexer = torch.LongTensor(np.arange(num_types)).unsqueeze(0)
    for minibatch in eval_data.get_batch():
        all_tensors = get_tensors_from_minibatch(minibatch, type_indexer, None, None, 0, config, test_bag_size, train = False)

        if config.encoder == "rnn_phrase":
            aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
        else:
            aux_data = all_tensors['position_embeddings']


        linking_data = [all_tensors['entity_candidates'], all_tensors['priors']] if config.linker_weight > 0 else None
        logit_dict = model(all_tensors['type_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], aux_data, test_bag_size, None, linking_data)

        if config.mode == 'typing':
            gold_ids = np.array(minibatch.data['gold_types']).reshape(-1, test_bag_size, num_types)[:, 0, :]
            scores = logit_dict['typing_logits'].data.cpu().numpy()
            predicted_ids = predict_typing(scores, gold_ids)
            _ap += AP(scores, gold_ids)
            accuracy += strict(predicted_ids)
            _len += len(gold_ids)
        else:
            predicted_ids = predict_linking(logit_dict['linking_logits'], all_tensors['entity_candidates']).data.cpu().numpy()
            accuracy += (minibatch.data['gold_ids'] == predicted_ids).sum()
            _len += len(minibatch.data['gold_ids'])



        if log_file is not None:
            write_to_file(f, minibatch, predicted_ids, scores, scores_all.data.cpu().numpy())


    if log_file is not None:
        f.close()

    print("accuracy: %5.4f" %(accuracy/float(_len)))
    if config.dataset == "figer" or config.mode == 'linking':
        return accuracy/float(_len)
    else:
        return _ap/float(_len)


def write_to_file(f, minibatch, predictions, scores, scores_all):
    sentences  = minibatch.data['context']
    gold_types = minibatch.data['gold_types']
    print(scores_all.shape)
    bag_size = len(sentences)/len(scores)
    for i in xrange(len(scores)):
        curr_entity   = minibatch.data['ent'][i*bag_size]
        curr_golds, _ = predictions[i]
        curr_gold_ids = map(lambda idx: inverse_type_dict[idx], curr_golds)
        sorted_scores = sorted(enumerate(scores[i]), key = lambda val : -1*val[1])
        predictions_curr = [ (inverse_type_dict[idx], score) for (idx, score) in sorted_scores ][:10]

        f.write("gold entity: %s\n" %curr_entity)
        f.write("gold types: %s\n" %(" ".join(curr_gold_ids)))
        f.write("Predicted gold types: %s\n" %(predictions_curr))
        f.write("------------------\n")
        for j in xrange(bag_size):
            curr_mention = " ".join([inverse_vocab_dict[_idx] for _idx in sentences[i*bag_size + j]])
            sorted_scores_curr = sorted(enumerate(scores_all[i,j]), key = lambda val : -1*val[1])
            predictions_curr_mention = [ (inverse_type_dict[idx], score) for (idx, score) in sorted_scores_curr][:10]
            f.write("%s\n" %curr_mention)
            f.write("mention predictions: %s\n" %(predictions_curr_mention))

        f.write("=================\n")



def filter_fb_types(type_dict, entity_type_dict):
    fb_types = [_type for _type in type_dict if not _type.startswith("Synset")]
    wordnet_types = [_type for _type in type_dict if _type.startswith("Synset")]

    # reorder types to make fb types appear first
    all_types = fb_types + wordnet_types

    orig_idx2type = {idx : _type for (_type, idx) in type_dict.iteritems()}
    type2idx = {_type : idx for (idx, _type) in enumerate(all_types)}


    # 1. filter out only fb types and 2. change fb IDs according to type2idx
    fb_entity_type_dict = {}
    for ent in entity_type_dict:
        curr = []
        for type_id in entity_type_dict[ent]:
            orig_type = orig_idx2type[type_id]
            if not orig_type.startswith("Synset"):
                curr.append(type2idx[orig_type])

        assert(len(curr) != 0)
        fb_entity_type_dict[ent] = set(curr) # easy to search


    return type2idx, fb_entity_type_dict, len(fb_types)

'''
Parse inputs
'''
def get_params():
    parser = argparse.ArgumentParser(description = 'Entity linker')
    parser.add_argument('-dataset', action="store", default="ACE", dest="dataset", type=str)
    parser.add_argument('-model_name', action="store", default="entity_linker", dest="model_name", type=str)
    parser.add_argument('-dropout', action="store", default=0.5, dest="dropout", type=float)
    parser.add_argument('-train', action="store", default=1, dest="train", type=int)
    parser.add_argument('-bag_size', action="store", default=10, dest="bag_size", type=int)
    parser.add_argument('-encoder', action="store", default="position_cnn", dest="encoder", type=str)
    parser.add_argument('-asymmetric', action = "store", default=0, dest="asymmetric", type=int)
    parser.add_argument('-struct_weight', action = "store", default = 0, dest="struct_weight", type=float)
    parser.add_argument('-linker_weight', action = "store", default = 0, dest = "linker_weight", type=float)
    parser.add_argument('-typing_weight', action = "store", default = 0, dest = "typing_weight", type=float)
    parser.add_argument('-mode', action = "store", default = "typing", dest = "mode", type=str)
    parser.add_argument('-bilinear_l2', action = "store", default = 0.0, dest = "bilinear_l2", type=float)
    parser.add_argument('-parent_sample_size', action = "store", default = 100, dest="parent_sample_size", type=int)
    parser.add_argument('-complex', action = "store", default = 0, dest = "complex", type=int)


    parser.add_argument('-base_dir', action="store", default="/iesl/canvas/smurty/epiKB", type=str)
    parser.add_argument('-lr', action="store", default=1e-3, type=float)
    parser.add_argument('-beta1', action="store", default=0.9, type=float)
    parser.add_argument('-beta2', action="store", default=0.999, type=float)
    parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
    parser.add_argument('-weight_decay', action="store", default=0.0, type=float)
    parser.add_argument('-save_model', action="store", default=1, type=int)
    parser.add_argument('-clip_val', action="store", default=5, type=int)
    parser.add_argument('-embedding_dim', action="store", default=300, type=int)
    parser.add_argument('-hidden_dim', action="store", default=150, type=int)
    parser.add_argument('-num_epochs', action="store", default=2, type=int)
    parser.add_argument('-kernel_width', action="store", default=5, type=int)
    parser.add_argument('-batch_size', action="store", default=256, type=int)
    parser.add_argument('-test_batch_size', action="store", default=1024, type=int)
    parser.add_argument('-take_frac', action="store", default=1.0, type=float)
    parser.add_argument('-use_transitive', action="store", default=1, type=int)


    opts = parser.parse_args(sys.argv[1:])
    for arg in vars(opts):
        print arg, getattr(opts, arg)
    return opts


def read_entities(filename, take_frac = 1.0):
    f = open(filename)
    entities = set()

    for line in f:
        line = line.strip().split("\t")
        entities.add(line[0])

    if take_frac != 1.0:
        take_idx = int(len(entities)*take_frac)
	return set(list(entities)[: take_idx])
    else:
        return entities


if __name__ == "__main__":

    opts = get_params()
    run_dir = os.getcwd()
    config_obj = Config_MIL(run_dir, opts)

    assert(config_obj.mode in ['typing', 'linking'])

    # === load in all the auxiliary data

    type_dict = joblib.load(config_obj.type_dict)
    entity_type_dict = joblib.load(config_obj.entity_type_dict)
    entity_dict  = joblib.load(config_obj.entity_dict)

    typenet_matrix = joblib.load(config_obj.typenet_matrix)


    type_dict, entity_type_dict, fb_type_size = filter_fb_types(type_dict, entity_type_dict)

    vocab_dict = joblib.load(config_obj.vocab_file)
    config_obj.vocab_size   = len(vocab_dict)
    config_obj.type_size    = len(type_dict)
    config_obj.entity_size  = len(entity_dict)

    config_obj.fb_type_size = fb_type_size

    inverse_vocab_dict = {idx: word for (word, idx) in vocab_dict.iteritems()}
    inverse_type_dict = {idx: word for (word, idx) in type_dict.iteritems()}
    inverse_entity_dict = {idx : word for (word, idx) in entity_dict.iteritems()}

    logger.info("\nNumber of words in vocab: %d\n" %config_obj.vocab_size)
    logger.info("\nNumber of total types in vocab: %d\n" %config_obj.type_size)
    logger.info("\nNumber of total entities in vocab: %d\n" %config_obj.entity_size)
    logger.info("\nNumber of fb types in vocab: %d\n" %config_obj.fb_type_size)

    pretrained_embeddings = get_trimmed_glove_vectors(config_obj.embedding_file)

    #=== now load in crosswikis
    if config_obj.linker_weight > 0:
        time_st = time.time() 
        alias_table = joblib.load(config_obj.cross_wikis_shelve)
        time_en = time.time()
        print("Time taken for reading alias table: %5.4f" %(time_en - time_st))

    else:
        alias_table = {}

    
    attribs = ['mention_representation', 'context', 'position_embeddings', 'gold_types', 'ent', 'st_ids', 'en_ids', 'entity_candidates', 'priors', 'gold_ids']

    if config_obj.encoder == "position_cnn":
        print("Using position embeddings with CNN")
        encoder = MentionEncoderCNNWithPosition(config_obj, pretrained_embeddings)
    elif config_obj.encoder == "basic":
        print("using regular CNNs")
        encoder = MentionEncoderCNN(config_obj, pretrained_embeddings)
    elif config_obj.encoder == "rnn_phrase":
        print("using a deep GRU and phrase embeddings")
        encoder = MentionEncoderRNN(config_obj, pretrained_embeddings)

    if config_obj.complex:
        model = MultiInstanceTyper_complex(config_obj, encoder)
    else:
        model   = MultiInstanceTyper(config_obj, encoder)

        
    load_model(model, config_obj)

    # === Define the training and dev data

    train_bags = joblib.load(config_obj.bag_file)
    dev_bags   = joblib.load(config_obj.bag_file)
    test_bags  = joblib.load(config_obj.bag_file)

    train_entities = read_entities(config_obj.train_file, config_obj.take_frac)
    dev_entities   = read_entities(config_obj.dev_file)
    test_entities  = read_entities(config_obj.test_file)


    train_data = MILDataset(train_bags, entity_type_dict, attribs, train_entities,vocab_dict, pretrained_embeddings, config_obj.encoder, config_obj.batch_size, config_obj.fb_type_size, MILtransformer, config_obj.bag_size, entity_dict, alias_table, True)
    dev_data   = MILDataset(dev_bags, entity_type_dict_test, attribs, dev_entities,  vocab_dict, pretrained_embeddings, config_obj.encoder, config_obj.test_batch_size, config_obj.fb_type_size, MILtransformer, config_obj.test_bag_size, entity_dict, alias_table, False)
    test_data  = MILDataset(test_bags, entity_type_dict_test, attribs, test_entities, vocab_dict, pretrained_embeddings, config_obj.encoder, config_obj.test_batch_size, config_obj.fb_type_size, MILtransformer, config_obj.test_bag_size, entity_dict, alias_table, False)


    if opts.train:
        logger.info("\n====================TRAINING STARTED====================\n")
        train(model, train_data, dev_data, test_data, config_obj, typenet_matrix)
    else:
        logger.info("\n====================EVALUATION STARTED====================\n")
        if config_obj.gpu:
            model.cuda()


        acc_test = evaluate(model, test_data, config_obj) #"results_test_%s.txt" %config_obj.model_file)
        acc_dev = evaluate(model, dev_data, config_obj)   #"results_dev_%s.txt" %config_obj.model_file)

        logger.info("\n: Dev accuracy %5.4f" %acc_dev)
        logger.info("\n: Test accuracy %5.4f" %acc_test)
