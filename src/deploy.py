'''
    Code for training/testing/evaluating/predicting a generic entity linker.
    TODO: implement checkpointing, logging
    UNK always maps to 0
'''


import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import *
from data_iterator import *
from config import Config
from build_data import *
from general_utils import *
import os
import torch.optim as optim


logging.basicConfig(stream = sys.stdout, level=logging.INFO)
logger = logging.getLogger("Mention Typing Basic")


def get_tensor_feed(left_contexts, right_contexts, on_gpu):
    '''
        converts to tensors and pads in a way that it can be fed into an entity linker
    '''

    padded_left_contexts, left_context_lens = pad_sequences(left_contexts, PAD, torch.LongTensor, on_gpu)
    padded_right_contexts, right_contexts_lens = pad_sequences(right_contexts, PAD, torch.LongTensor, on_gpu)

    return (padded_left_contexts, left_context_lens, padded_right_contexts, right_contexts_lens)



# == Helpers for creating the loss function and performing gradient descent, and some data stat functions



def loss(scores, entity_candidates, entity_candidate_lens, gold_type_id, criterion):
    '''
        scores is a batch_size x num_candidates*num_types tensor of logits. where the slice from e*num_types to (e+1)*num_types refers to the logits for the e-th candidate and all types
        entity_candidate_lens is the actual number of candidates per example
    '''


    batch_size = scores.size()[0]
    num_candidates = entity_candidates.size()[-1]
    num_types = scores.size()[-1]/num_candidates

    targets = num_types*(entity_candidate_lens - 1) + gold_type_id # the gold entity is the last one.
    #print(target.shape())
    return criterion(scores, targets)




# == The main training/testing/prediction functions

def get_tensors_from_minibatch(minibatch, type_indexer, type_indexer_struct, typenet_matrix, num_iter, config, train = True):
    mention_representation = np.array(minibatch.data['mention_representation']) # (batch_size, embedding_dim)
    batch_size = len(mention_representation)
    # possible mention types
    type_idx = type_indexer.repeat(batch_size, 1) #(batch_size, num_fb_types)
    gold_types = torch.FloatTensor(minibatch.data['gold_types']) #(batch_size, num_types)
    mention_representation = torch.FloatTensor(mention_representation)

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

        structure_parents_gold = torch.FloatTensor(typenet_matrix[type_curr_child]) # get parent of children sampled
        structure_parents_candidates  = type_indexer_struct.repeat(batch_size_types, 1)
        structure_children = torch.LongTensor(type_curr_child.reshape(batch_size_types, 1)) # convert children sampled to tensors

        all_tensors['structure_parents_gold'] = structure_parents_gold
        all_tensors['structure_parent_candidates'] = structure_parents_candidates
        all_tensors['structure_children'] = structure_children

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


    # == feed to mention typer and get scores ==
    sentences , _ = pad_sequences(minibatch.data['context'], PAD, torch.LongTensor, config.gpu)
    position_embeddings, _ = pad_sequences(minibatch.data['position_embeddings'], PAD, torch.LongTensor, config.gpu, pos = True)


    all_tensors['sentences'] = sentences
    all_tensors['position_embeddings'] = position_embeddings

    return all_tensors



def train(model, train, dev, test, config, typenet_matrix):
    '''
    :param model: should adhere to the API of an entity linker
    :param dataset_files: a list of files that constitute the training data
    :param dev: the dev-set data
    '''

    # define the criterion for computing loss : currently only supports cross entropy

    criterion_linking = nn.CrossEntropyLoss()
    criterion_typing = nn.BCEWithLogitsLoss()


    # define the optimizer : currently only supports Adam
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=config.lr, eps=config.epsilon,
                           weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))

    # transfer to GPU
    if config.gpu:
        model.cuda()


    best_accuracy, _, _ = evaluate(model, dev, config)

    num_types = config.type_size
    type_indexer = torch.LongTensor(np.arange(num_types)).unsqueeze(0)
    type_indexer_struct = torch.LongTensor(np.arange(num_types)).unsqueeze(0) #(1, num_types) for indexing into type embeddings array

    num_iter = 0

    logger.info("\n==== Initial accuracy: %5.4f\n" %best_accuracy)
    for epoch in xrange(config.num_epochs):
        logger.info("\n=====Epoch Number: %d =====\n" %(epoch+1))

        train.shuffle()
        for minibatch in train.get_batch():
            all_tensors = get_tensors_from_minibatch(minibatch, type_indexer, type_indexer_struct, typenet_matrix, num_iter, config)

            if config.encoder == "rnn_phrase":
                aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
            else:
                aux_data = all_tensors['position_embeddings']

            # set to train

            feature_data = all_tensors['features'] if config.features else None

            model.train()
            if config.struct_weight > 0:
                struct_data = [all_tensors['structure_children'], all_tensors['structure_parent_candidates']]
                scores = model(all_tensors['type_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], aux_data, feature_data, struct_data)
                typing_loss = criterion_typing(scores[0], all_tensors['gold_types'])
                structure_loss = criterion_typing(scores[1], all_tensors['structure_parents_gold'])
                train_loss = typing_loss + config.struct_weight*structure_loss
                step(model, train_loss, optimizer, config.clip_val)
            else:
                scores = model(all_tensors['type_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], aux_data, feature_data)
                train_loss = criterion_typing(scores, all_tensors['gold_types'])
                step(model, train_loss, optimizer, config.clip_val)

            # == using the scores, calculate the loss.
            if num_iter % 10 == 0:
                logger.info("\n=====Loss: %5.4f Step: %d ======\n" %(train_loss.data[0], num_iter))

            if num_iter % 250 == 0:
                curr_accuracy, _, _ = evaluate(model, test, config)
                logger.info("\n==== Accuracy on dev after %d iterations:\t %5.4f\n" % (num_iter, curr_accuracy))
                if curr_accuracy > best_accuracy:
                    best_accuracy = curr_accuracy
                    test_accuracy, micro_f1, macro_f1 = evaluate(model, test, config)
                    logger.info("\n==== Accuracy on test after %d iterations:\t %5.4f\n" % (num_iter, test_accuracy))
                    logger.info("\n==== Micro F1 on test after %d iterations:\t %5.4f\n" % (num_iter, micro_f1))
                    logger.info("\n==== Macro F1 on test after %d iterations:\t %5.4f\n" % (num_iter, macro_f1))
                    if config.save_model:
                        save_model(model, config)

            num_iter += 1


    # load the best model
    load_model(model, config)

    accuracy_test, micro_f1, macro_f1 = evaluate(model, test, config)

    logger.info("\n==== Training Finished! =====\n")
    logger.info("\n==== Accuracy on Test: %5.4f \n" %accuracy_test)
    logger.info("\n==== Micro F1 on Test: %5.4f\n" % (num_iter, micro_f1))
    logger.info("\n==== Macro F1 on Test: %5.4f\n" % (num_iter, macro_f1))




def predict(scores, gold_ids, threshold=.5):
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
    num_types = model.config.type_size # only index into freebase partition

    model.eval()
    if log_file is not None:
        f = open(log_file, "w")

    scorer_obj = Scorer()

    type_indexer = torch.LongTensor(np.arange(num_types)).unsqueeze(0)
    for minibatch in eval_data.get_batch():
        all_tensors = get_tensors_from_minibatch(minibatch, type_indexer, None, None, 0, config, train = False)
        feature_data = all_tensors['features'] if config.features else None


        if config.encoder == "rnn_phrase":
            aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
        else:
            aux_data = all_tensors['position_embeddings']


        scores = model(all_tensors['type_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], aux_data, feature_data)
        gold_ids = np.array(minibatch.data['gold_types'])

        scores = scores.data.cpu().numpy()

        predicted_ids = predict(scores, gold_ids)
        scorer_obj.run(predicted_ids)
        if log_file is not None:
            write_to_file(f, minibatch, predicted_ids, scores, scores_all.data.cpu().numpy())


    if log_file is not None:
        f.close()

    if config.dataset == "figer":
        return scorer_obj.get_scores() 
    else:
        return _ap/float(_len)





# == Helper function for writing predictions to a file

def write_to_file(f, minibatch, predicted_ids, probabilities_typing):

    sentence = minibatch.data['context']
    for i in xrange(len(sentence)):
        curr_sent = map(lambda _id: inverse_vocab_dict[_id], sentence[i])
        curr_mention = " ".join(curr_sent)

        gold_curr, predicted_curr = predicted_ids[i]
        curr_violations = [(val, j) for (j, val) in enumerate(probabilities_typing[i])]
        curr_violations = sorted(curr_violations, reverse=True)

        gold_curr = " ".join(map(lambda _id : inverse_type_dict[_id], gold_curr))
        predicted_curr = " ".join(map(lambda _id : inverse_type_dict[_id], predicted_curr))

        mention_types = " ".join(map(lambda _id: inverse_type_dict[_id[1]], curr_violations[:10]))

        readable_probabilites = [(probability, inverse_type_dict[_id]) for (_id, probability) in enumerate(probabilities_typing[i])]
        readable_probabilites = sorted(readable_probabilites, reverse=True, key=lambda tup: tup[0])

        f.write("\nmention: %s\n" %curr_mention)
        # f.write("mention types: %s\n" %mention_types)
        f.write("actual: %s\n" %(gold_curr))
        # f.write("predictions: %s, actual: %s\n" %(predicted_curr, gold_curr))
        f.write("%s\n" %readable_probabilites[:20])

        f.write("\n")


'''
Parse inputs
'''
def get_params():
    parser = argparse.ArgumentParser(description = 'Entity linker')
    parser.add_argument('-dataset', action="store", default="ACE", dest="dataset", type=str)
    parser.add_argument('-model_name', action="store", default="entity_linker", dest="model_name", type=str)
    
    parser.add_argument('-dropout', action="store", default=0.5, dest="dropout", type=float)
    parser.add_argument('-bag_size', action="store", default=10, dest="bag_size", type=int)
    parser.add_argument('-asymmetric', action = "store", default=0, dest="asymmetric", type=int)
    parser.add_argument('-struct_weight', action = "store", default = 0, dest="struct_weight", type=float)
    parser.add_argument('-linker_weight', action = "store", default = 0, dest = "linker_weight", type=float)
    parser.add_argument('-typing_weight', action = "store", default = 0, dest = "typing_weight", type=float)
    parser.add_argument('-mode', action = "store", default = "typing", dest = "mode", type=str)
    parser.add_argument('-bilinear_l2', action = "store", default = 0.0, dest = "bilinear_l2", type=float)
    parser.add_argument('-parent_sample_size', action = "store", default = 100, dest="parent_sample_size", type=int)
    parser.add_argument('-complex', action = "store", default = 0, dest = "complex", type=int)
    parser.add_argument('-features', action = "store", default = 0, dest = "features", type=int)


    parser.add_argument('-dropout_bl', default=0.0, type=float)
    parser.add_argument('-train', action="store", default=1, dest="train", type=int)
    parser.add_argument('-dist_func', action="store", default='dot', dest="distance_func", type=str)
    parser.add_argument('-encoder', action="store", default='position_cnn', dest="encoder", type=str)

    parser.add_argument('-base_dir', action="store", default="/iesl/canvas/smurty/epiKB", type=str)
    parser.add_argument('-lr', action="store", default=1e-3, type=float)
    parser.add_argument('-beta1', action="store", default=0.9, type=float)
    parser.add_argument('-beta2', action="store", default=0.999, type=float)
    parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
    parser.add_argument('-weight_decay', action="store", default=0.0, type=float)
    parser.add_argument('-margin', action="store", default=1.0, type=float)
    parser.add_argument('-save_model', action="store", default=1, type=int)
    parser.add_argument('-clip_val', action="store", default=5, type=int)
    parser.add_argument('-embedding_dim', action="store", default=300, type=int)
    parser.add_argument('-hidden_dim', action="store", default=150, type=int)
    parser.add_argument('-num_epochs', action="store", default=2, type=int)
    parser.add_argument('-kernel_width', action="store", default=5, type=int)
    parser.add_argument('-batch_size', action="store", default=256, type=int)
    parser.add_argument('-repeats', action="store", default=0, type=int)


    opts = parser.parse_args(sys.argv[1:])
    for arg in vars(opts):
        print arg, getattr(opts, arg)
    return opts

def read_file(filename):
    data = []
    with gzip.open(filename) as f:
        for line in f:
            data.append(line)

    return data

if __name__ == "__main__":

    opts = get_params()
    run_dir = os.getcwd()
    config_obj = Config(run_dir, opts)

    # === load in all the auxiliary data

    entity_dict = joblib.load(config_obj.entity_file)
    type_dict = joblib.load(config_obj.type_file)
    entity_type_dict = joblib.load(config_obj.entity_type_file)
    typenet_matrix = np.ones((len(type_dict), len(type_dict)))

    vocab_dict = joblib.load(config_obj.vocab_file)
    config_obj.vocab_size = len(vocab_dict)
    config_obj.entity_size = len(entity_dict)
    config_obj.type_size = len(type_dict)


    feature_dict = dict([(idx, feat) for (idx, feat, _hash) in open(self.config.feature_file)])
    config_obj.feature_size = len(feature_dict)


    inverse_entity_dict = {idx : word for (word, idx) in entity_dict.iteritems()}
    inverse_vocab_dict = {idx: word for (word, idx) in vocab_dict.iteritems()}
    inverse_type_dict = {idx: word for (word, idx) in type_dict.iteritems()}

    logger.info("\nNumber of entities: %d\n" %len(entity_dict))
    logger.info("\nNumber of words in vocab: %d\n" %len(vocab_dict))
    logger.info("\nNumber of types in vocab: %d\n" %len(type_dict))



    pretrained_embeddings = get_trimmed_glove_vectors(config_obj.embedding_file)

    # === now load in crosswikis
    #crosswikis_obj = CrossWikis(config_obj.cross_wikis_shelve)
    #crosswikis_obj.open_shelve()

    # === Define the entity linker and load it in



    attribs = ['mention_representation', 'context', 'position_embeddings', 'gold_types', 'ent', 'st_ids', 'en_ids']

    if config_obj.features:
        attribs.append('features')


    if config_obj.encoder == "position_cnn":
        print("Using position embeddings with CNN")
        encoder = MentionEncoderCNNWithPosition(config_obj, pretrained_embeddings)
    elif config_obj.encoder == "basic":
        print("using regular CNNs")
        encoder = MentionEncoderCNN(config_obj, pretrained_embeddings)
    elif config_obj.encoder == "rnn_phrase":
        print("using a deep GRU and phrase embeddings")
        encoder = MentionEncoderRNN(config_obj, pretrained_embeddings)


    if config_obj.distance_func == "order":
        model = BasicMentionTyperOrder(config_obj, encoder)
    elif config_obj.complex:
        print("Using complex")
        model = ComplexMentionTyperDot(config_obj, encoder)
    else:
        print("Using real")
        model = BasicMentionTyperDot(config_obj, encoder)

    load_model(model, config_obj)

    # === Define the training and dev data
    train_files = read_file(glob.glob("%s/*.gz" %config_obj.train_file)[0])
    dev_files = read_file(glob.glob("%s/*.gz" %config_obj.dev_file)[0])
    test_files = read_file(glob.glob("%s/*.gz" %config_obj.test_file)[0])


    if opts.dataset == "figer":
        train_data = TypingDataset(train_files, attribs, entity_dict, vocab_dict, feature_dict, pretrained_embeddings, config_obj.batch_size , transform_sentence_wiki_typing, True, type_dict,  config_obj.type_size, config_obj.encoder)
        dev_data   = TypingDataset(dev_files, attribs, entity_dict, vocab_dict, feature_dict, pretrained_embeddings, config_obj.batch_size, transform_sentence_wiki_typing, False, type_dict,  config_obj.type_size, config_obj.encoder)
        test_data  = TypingDataset(test_files, attribs, entity_dict, vocab_dict, feature_dict, pretrained_embeddings, config_obj.batch_size, transform_sentence_wiki_typing, False, type_dict,  config_obj.type_size, config_obj.encoder)

    elif opts.dataset == "typenet":
        train_data = TypingDataset(train_files, attribs, entity_dict, vocab_dict, feature_dict, pretrained_embeddings, config_obj.batch_size, transform_sentence_wiki, True, entity_type_dict,  config_obj.type_size, config_obj.encoder)
        dev_data   = TypingDataset(dev_files, attribs, entity_dict, vocab_dict, feature_dict, pretrained_embeddings, config_obj.batch_size,   transform_sentence_aida, False, entity_type_dict,  config_obj.type_size, config_obj.encoder)
        test_data  = TypingDataset(test_files, attribs, entity_dict, vocab_dict, feature_dict, pretrained_embeddings, config_obj.batch_size,  transform_sentence_aida, False, entity_type_dict,  config_obj.type_size, config_obj.encoder)


    if opts.train:
        logger.info("\n====================TRAINING STARTED====================\n")
        train(model, train_data, dev_data, test_data, config_obj, typenet_matrix)
    else:
        logger.info("\n====================EVALUATION STARTED====================\n")
        if config_obj.gpu:
            model.cuda()
        acc_test = evaluate(model, test_data, config_obj, add_priors=True, log_file = "results_test.txt")
        acc_dev = evaluate(model, dev_data, config_obj, add_priors=True, log_file = "results_dev.txt")
        logger.info("\n: Dev accuracy %5.4f" %acc_dev)
        logger.info("\n: Test accuracy %5.4f" %acc_test)
