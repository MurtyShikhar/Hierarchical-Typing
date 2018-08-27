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
logger = logging.getLogger("Entity Linker")

#TODO: Typing has not been yet added.

# == The main training/testing/prediction functions


def get_tensors_from_minibatch(minibatch, num_iter, config, entity_hierarchy_linked_list, train ):
    mention_representation = np.array(minibatch.data['mention_representation']) # (batch_size, embedding_dim)
    batch_size = len(mention_representation)

    mention_representation = torch.FloatTensor(mention_representation)
    all_tensors = {'mention_representation' : mention_representation}


    if config.struct_weight > 0 and train:
        # ========= STRUCTURE LOSS TENSORS ===========
        num_entities = config.entity_size
        st = (num_iter*batch_size) % num_entities
        en = (st + batch_size) % num_entities

        if en <= st:
            sampled_child_nodes = range(st, num_entities) + range(0, en)
        else:
            sampled_child_nodes = range(st, en)

        batch_size_struct   = len(sampled_child_nodes)
        parent_candidates   = []

        parent_candidate_labels = []
        for node in sampled_child_nodes:
            curr_labels = []
            parent_candidates_curr = np.random.choice(num_entities, config.parent_sample_size, replace = False) # choose a random set of parent links
            for parent_candidate in parent_candidates_curr:
                if node not in entity_hierarchy_linked_list:
                    curr_labels.append(0)
                elif parent_candidate in entity_hierarchy_linked_list[node]:
                    curr_labels.append(1)
                else:
                    curr_labels.append(0)

            parent_candidate_labels.append(curr_labels)
            parent_candidates.append(parent_candidates_curr)

        all_tensors['structure_parents_gold'] = torch.FloatTensor(parent_candidate_labels) # (batch_size_struct, config.parent_sample_size)
        all_tensors['structure_parent_candidates'] = torch.LongTensor(parent_candidates)   # (batch_size_struct, config.parent_sample_size)
        all_tensors['structure_children']  = torch.LongTensor(sampled_child_nodes).view(batch_size_struct, 1) # (batch_size_struct, 1)


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

    all_tensors['sentences'] , _ = pad_sequences(minibatch.data['context'], PAD, torch.LongTensor, config.gpu)
    all_tensors['position_embeddings'], _ = pad_sequences(minibatch.data['position_embeddings'], PAD, torch.LongTensor, config.gpu, pos = True)
    all_tensors['entity_candidates'], all_tensors['entity_candidate_lens'] = \
        pad_sequences(minibatch.data['entity_candidates'], PAD, torch.LongTensor, config.gpu) #(batch_size, 100)

    priors_padded, _ = pad_sequences(minibatch.data['priors'], 0.0, torch.FloatTensor, config.gpu)
    all_tensors['priors'] = priors_padded
    return all_tensors

def train(model, train, dev, test, config, entity_hierarchy_linked_list, entity_hierarchy_linked_list_orig):

    # define the criterion for computing loss : currently only supports cross entropy
    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion_multilabel = nn.BCEWithLogitsLoss()


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


    best_accuracy = evaluate(model, dev, config, entity_hierarchy_linked_list_orig)
    logger.info("\n==== Initial accuracy: %5.4f\n" %best_accuracy)

    num_iter = 0
    patience = 0
    for epoch in xrange(config.num_epochs):
        logger.info("\n=====Epoch Number: %d =====\n" %(epoch+1))
        for minibatch in train.get_batch():
            all_tensors = get_tensors_from_minibatch(minibatch, num_iter, config, entity_hierarchy_linked_list, True)

            if config.encoder == "rnn_phrase":
                aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
            else:
                aux_data = all_tensors['position_embeddings']

            # set back to train mode
            model.train()

            if config.struct_weight > 0:
                struct_data = [all_tensors['structure_children'], all_tensors['structure_parent_candidates']]
            else:
                struct_data = None


            if config.typing_weight > 0:
                type_data = all_tensors['typing_data']
            else:
                type_data = None

            scores = model(all_tensors['entity_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], all_tensors['priors'], aux_data, struct_data, type_data)
            linking_loss   = loss_criterion(scores['linking_logits'], all_tensors['entity_candidate_lens'] - 1)
            structure_loss = loss_criterion_multilabel(scores['structure_logits'], all_tensors['structure_parents_gold']) if config.struct_weight > 0 else 0.0
            typing_loss    = loss_criterion_multilabel(scores['typing_logits'], all_tensors['typing_gold']) if config.typing_weight > 0 else 0.0

            train_loss = linking_loss + config.struct_weight*structure_loss + config.struct_weight*typing_loss
            step(model, train_loss, optimizer, config.clip_val)

            if num_iter % 10 == 0:
                logger.info("\n=====Loss: %5.4f Step: %d ======\n" %(train_loss.data[0], num_iter))

            if (num_iter % 500 == 0) and num_iter > 1500:
                curr_accuracy = evaluate(model, dev, config, entity_hierarchy_linked_list_orig)
                logger.info("\n==== Accuracy on dev after %d iterations:\t %5.4f\n" % (num_iter, curr_accuracy))
                if curr_accuracy > best_accuracy:
                    best_accuracy = curr_accuracy
                    test_accuracy = evaluate(model, test, config, entity_hierarchy_linked_list_orig) # "result_log/acc_test_%s.txt" %config.model_file)
                    logger.info("\n==== Accuracy on test after %d iterations:\t %5.4f\n" % (num_iter, test_accuracy))
                    if config.save_model:
                        save_model(model, config)
                    patience = 0
                else:
                    patience += 1
                    if patience > 10000:
                        break

            num_iter += 1


    # load the best model
    load_model(model, config)

    accuracy_test = evaluate(model, test, config, entity_hierarchy_linked_list_orig)

    logger.info("\n===== Training Finished! =====\n")
    logger.info("\n===== Accuracy on Test: %5.4f \n" %accuracy_test)


def predict(scores, entity_candidates):

    # scores is a batch_size x num_candidates tensor of logits.

    # == find the argmax and use that to index into entity_candidates ===
    _, idx = scores.max(dim = -1)
    return entity_candidates.gather(1, idx.unsqueeze(1)).squeeze(1)




def evaluate(model, eval_data, config, entity_hierarchy_linked_list, log_file = None):
    accuracy = 0.0
    _len = 0
    hierarchical_accuracy = 0.0

    model.eval()
    if log_file is not None:
        f = open(log_file, "w")


    for minibatch in eval_data.get_batch():
        all_tensors = get_tensors_from_minibatch(minibatch, 0, config, None,  False)

        if config.encoder == "rnn_phrase":
            aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
        else:
            aux_data = all_tensors['position_embeddings']

        scores = model(all_tensors['entity_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], all_tensors['priors'] , aux_data)
        predicted_ids = predict(scores['linking_logits'], all_tensors['entity_candidates']).data.cpu().numpy()

        hierarchical_accuracy += get_hierarchical_accuracy(minibatch.data['gold_ids'], predicted_ids, entity_hierarchy_linked_list)
        accuracy += (minibatch.data['gold_ids'] == predicted_ids).sum()
        _len += len(minibatch.data['gold_ids'])

        if log_file is not None:
            write_to_file(f, minibatch, predicted_ids)


    if log_file is not None:
        f.close()

    logger.info("Hierarchical Accuracy: %5.4f\n" % (hierarchical_accuracy / float(_len)))
    return (accuracy/float(_len))


def write_to_file(f, minibatch, predicted_ids):
    _len = len(minibatch.data['gold_ids'])
    gold_ids = minibatch.data['gold_ids']
    sentences = minibatch.data['context']
    batch_candidates = minibatch.data['entity_candidates']

    def process(j, idx, st, en):
        if j == st and j == en:
            return "<target> %s </target>" %(inverse_vocab_dict[idx])
        if j == st:
            return "<target> %s" %(inverse_vocab_dict[idx])
        elif j == en:
            return "%s </target>" %(inverse_vocab_dict[idx])
        else:
            return inverse_vocab_dict[idx]

    for i in xrange(_len):
        batch_candidates_curr = set(batch_candidates[i])
        position_curr = minibatch.data['position_embeddings'][i]

        st = minibatch.data['st_ids'][i][0]
        en = minibatch.data['en_ids'][i][0]

        curr_sentence = " ".join(process(j,idx, st, en) for j,idx in enumerate(sentences[i]))
        verdict = "CORRECT" if gold_ids[i] == predicted_ids[i] else "WRONG"
        prediction = "%s,%s" %(canonical_inverse_entity_dict[predicted_ids[i]], inverse_entity_dict[predicted_ids[i]])
        actual = "%s,%s" %(canonical_inverse_entity_dict[gold_ids[i]], inverse_entity_dict[gold_ids[i]])
        candidates = " |".join("%s,%s" %(canonical_inverse_entity_dict[idx], inverse_entity_dict[idx]) for idx in batch_candidates_curr)         
        f.write("%s\n" %curr_sentence)
        f.write("(%s,%s)" %(st,en))
        f.write("CANDIDATES: %s\n" %candidates)
        f.write("PREDICTION: %s\n" %prediction)
        f.write("ACTUAL: %s\n" %actual)
        f.write("%s\n" %verdict)
        f.write("\n")

    


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
    parser.add_argument('-struct_weight', action = "store", default = 0, dest="struct_weight", type=float)
    parser.add_argument('-typing_weight', action = "store", default = 0, dest="typing_weight", type=float)
    parser.add_argument('-multilabel_linking', action = "store", default = 0, dest = "multilabel_linking", type=int)


    parser.add_argument('-parent_sample_size', action = "store", default = 100, dest="parent_sample_size", type=int)
    parser.add_argument('-bilinear_l2', action = "store", default = 0.0005, dest = "bilinear_l2", type=float)
    parser.add_argument('-bilinear_bias', action = "store", default = 0, dest = "bilinear_bias", type=int)
    parser.add_argument('-complex', action = "store", default = 0, dest = "complex", type=int)
    parser.add_argument('-asymmetric', action = "store", default = 1, dest = "asymmetric", type=int)
    parser.add_argument('-learn_graph', action = "store", default = 0, dest = "learn_graph", type=int)
    parser.add_argument('-priors', action = "store", default = 1, dest = "priors", type=int)



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
    parser.add_argument('-take_frac', action="store", default=1.0, type=float)
    parser.add_argument('-use_transitive', action="store", default=1, type=int)


    opts = parser.parse_args(sys.argv[1:])
    for arg in vars(opts):
        print arg, getattr(opts, arg)
    return opts




if __name__ == "__main__":

    opts = get_params()
    run_dir = os.getcwd()
    config_obj = Config(run_dir, opts)

    train_lines = joblib.load("meta_data_processed/meta_train_transitive.joblib")
    dev_lines   = joblib.load("meta_data_processed/meta_dev.joblib")
    test_lines  = joblib.load("meta_data_processed/meta_test.joblib")
    print("size of training data: %5.4f" %len(train_lines))

    # === load in all the auxiliary data
    vocab_dict  = joblib.load(config_obj.vocab_file)
    entity_dict = joblib.load(config_obj.entity_file)
    type_dict   = {} #joblib.load(config_obj.type_file)

    entity_hierarchy_linked_list = joblib.load(config_obj.hierarchy) # a dict mapping entity IDs to a set of their parent entity IDs (transitive closure included)
    entity_hierarchy_linked_list_orig = joblib.load(config_obj.hierarchy_orig) # a dict mapping entity IDs to a set of their parent entity IDs (without transitive closure)


    config_obj.vocab_size   = len(vocab_dict)
    config_obj.entity_size  = len(entity_dict)
    config_obj.type_size    = len(type_dict)

    inverse_vocab_dict = {idx: word for (word, idx) in vocab_dict.iteritems()}

    canonical = dict(tuple(line.strip().split("\t")) for line in gzip.open("meta_data_processed/cui_name_map.gz"))

    inverse_entity_dict = {idx : ent for (ent, idx) in entity_dict.iteritems()}
    canonical_inverse_entity_dict = {idx : canonical[ent] if ent in canonical else '<None>' for (ent, idx) in entity_dict.iteritems()}
    inverse_type_dict  = {idx : _type for (_type, idx) in type_dict.iteritems()}

    logger.info("\nNumber of words in vocab: %d\n" %config_obj.vocab_size)
    logger.info("\nNumber of entities in vocab: %d\n" %config_obj.entity_size)
    logger.info("\nNumber of types in vocab: %d\n" %config_obj.type_size)

    pretrained_embeddings = get_trimmed_glove_vectors(config_obj.embedding_file)

    # === now load in crosswikis
    alias_table = joblib.load(config_obj.cross_wikis_shelve)

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
        model = Linker_complex(config_obj, encoder)
    else:
        model = Linker_separate(config_obj, encoder)

        
    load_model(model, config_obj)


    
    ATTRIBS = ['mention_representation', 'context', 'position_embeddings', 'st_ids', 'en_ids', 'entity_candidates', 'gold_ids', 'priors']
    train_data = LinkingDataset(train_lines, ATTRIBS, entity_dict, vocab_dict, type_dict, alias_table, pretrained_embeddings, config_obj.batch_size,UMLS_transformer,True, config_obj.encoder)
    dev_data   = LinkingDataset(dev_lines, ATTRIBS, entity_dict,   vocab_dict, type_dict, alias_table, pretrained_embeddings, config_obj.batch_size,UMLS_transformer,False,config_obj.encoder)
    test_data  = LinkingDataset(test_lines, ATTRIBS, entity_dict,  vocab_dict, type_dict, alias_table, pretrained_embeddings, config_obj.batch_size,UMLS_transformer,False,config_obj.encoder)



    if opts.train:
        logger.info("\n====================TRAINING STARTED====================\n")
        train(model, train_data, dev_data, test_data, config_obj, entity_hierarchy_linked_list, entity_hierarchy_linked_list_orig)
    else:
        logger.info("\n====================EVALUATION STARTED====================\n")
        if config_obj.gpu:
            model.cuda()

        acc_test = evaluate(model, test_data, config_obj, entity_hierarchy_linked_list_orig, "results_test_%s.txt" %config_obj.model_file)
        acc_dev = evaluate(model, dev_data, config_obj, entity_hierarchy_linked_list_orig, "results_dev_%s.txt" %config_obj.model_file)

        logger.info("\n: Dev accuracy %5.4f" %acc_dev)
        logger.info("\n: Test accuracy %5.4f" %acc_test)
