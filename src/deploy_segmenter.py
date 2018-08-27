'''
    Code for training/testing/evaluating/predicting a generic entity linker.
    UNK always maps to 0
    Predictions are made only for freebase types, and hence, for ease we map all freebase types for ids first, and then map wordnet types
'''

import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import *
from model_pointer import *
from data_iterator import *
from build_data import *
from general_utils import *
from visualise import *
import os
import torch.optim as optim


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("Mention Typing Basic")

# == The main training/testing/prediction functions


def get_tensors_from_minibatch(minibatch, config):
    sentences, sentence_lengths = pad_sequences(minibatch.data['context'], PAD, torch.LongTensor, config.gpu) #(N, W)
    gold_ids,  _ = pad_sequences(minibatch.data['gold_ids'], 2, torch.LongTensor, config.gpu)


    all_tensors = {'context': sentences, 'gold_ids' : gold_ids}
    if config.segmenter == 'pointer':
        max_sent_len = sentences.size()[1]
        gold_ids_pointer, _ = pad_sequences(minibatch.data['gold_ids_pointer'], max_sent_len, torch.LongTensor, config.gpu)
        all_tensors['gold_ids_pointer'] = gold_ids_pointer[:,:2].contiguous() #(predict just the first mention)
    else:
        all_tensors['gold_ids_pointer'] = None

    all_tensors['context_lens'] = Variable(torch.LongTensor(sentence_lengths).cuda()).view(-1,1) if config.gpu else Variable(torch.LongTensor(sentence_lengths)).view(-1,1)

    if config.char_level:
        char_ids, char_lens = pad2d(minibatch.data['char_ids'], PAD, torch.LongTensor, config.gpu)
        all_tensors['char_ids'] = char_ids
        all_tensors['char_lens'] = char_lens

    else:
        all_tensors['char_ids']  = None
        all_tensors['char_lens'] = None

    return all_tensors


def train(model, train, dev, test, config):
    # define the criterion for computing loss : currently only supports cross entropy
    loss_criterion = nn.CrossEntropyLoss()

    # define the optimizer : currently only supports Adam
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=config.lr, eps=config.epsilon,
                           weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))

    # transfer to GPU
    if config.gpu:
        model.cuda()

    best_accuracy, best_f1 = (0,0) #evaluate(model, dev, config)
    logger.info("\n==== Initial accuracy: %5.4f, Initial f1: %5.4f \n" %(best_accuracy, best_f1))

    num_iter = 0
    for epoch in xrange(config.num_epochs):
        logger.info("\n=====Epoch Number: %d =====\n" % (epoch + 1))
        for minibatch in train.get_batch():
            all_tensors = get_tensors_from_minibatch(minibatch, config)

            # set back to train mode
            model.train()
            scores = model(all_tensors['context'], all_tensors['context_lens'], all_tensors['char_ids'], all_tensors['char_lens'], all_tensors['gold_ids_pointer'])


            if config.segmenter == 'pointer':
                train_loss = loss_criterion(scores, all_tensors['gold_ids_pointer'].view(-1))
                batch_size = all_tensors['context'].size()[0]
                scores = scores.view(batch_size,2,-1)
                _, preds_1 = scores[:,0,:].max(dim=-1)
                _, preds_2 = scores[:,1,:].max(dim=-1)
                constraint_loss= (preds_1.float() - preds_2.float()).clamp(min=0).mean()
                train_loss += constraint_loss

            else:
                train_loss = loss_criterion(scores, all_tensors['gold_ids'].view(-1))


            step(model, train_loss, optimizer, config.clip_val)

            if num_iter % 10 == 0:
                logger.info("\n=====Loss: %5.4f, Constraint Loss: %5.4f Step: %d ======\n" % (train_loss.data[0], constraint_loss.data[0], num_iter))

            if num_iter % 100 == 0:
                curr_accuracy, curr_f1 = evaluate(model, dev, config)
                logger.info("\n==== Accuracy on dev after %d iterations:\t %5.4f | F1 on dev: %5.4f\n"
                            % (num_iter, curr_accuracy, curr_f1))
                if curr_f1 > best_f1:
                    best_f1 = curr_f1

                    if config.save_model:
                        save_model(model, config)
                    else:
                        test_accuracy, test_f1 = evaluate(model, test, config)  # "acc_test_%s.txt" %config.model_file)
                        logger.info("\n==== Accuracy on test after %d iterations:\t %5.4f | F1 on test: %5.4f\n" %
                                (num_iter, test_accuracy, test_f1))

            num_iter += 1

    # load the best model
    load_model(model, config)

    accuracy_test, test_f1 = evaluate(model, test, config)

    logger.info("\n===== Training Finished! =====\n")
    logger.info("\n===== Accuracy on Test: %5.4f | F1 on test: %5.4f\n" % (accuracy_test, test_f1))




def evaluate(model, eval_data, config, log_file=None):
    accuracy = 0.0
    _len = 0

    correct_preds, total_correct, total_preds = 0., 0., 0.

    model.eval()
    if log_file is not None:
        f = open(log_file, "w")

    for minibatch in eval_data.get_batch():
        all_tensors = get_tensors_from_minibatch(minibatch, config)
        batch_size, max_sent_size = all_tensors['context'].size()
        scores = model(all_tensors['context'], all_tensors['context_lens'], all_tensors['char_ids'], all_tensors['char_lens'], all_tensors['gold_ids_pointer']).data.cpu().numpy()

        if config.segmenter == 'pointer':
            # scores is (N*k, W+1) where W is the max sent size
            predicted_ids = scores.argmax(-1).reshape(batch_size, -1)  #(N, k)
        else:
            #scores is (N*W, 3) where W is the max size
            predicted_ids = scores.argmax(-1).reshape(batch_size, -1)


        gold_ids = all_tensors['gold_ids_pointer'].data.cpu().numpy()

        sent_lens = all_tensors['context_lens'].data.cpu().numpy().reshape(-1)

        chunks_predicted_batch = []
        chunks_gold_batch      = []
        # for calculating F1 and accuracy
        for prediction, gold_id, sent_len in zip(predicted_ids, gold_ids, sent_lens):
            # only evaluate on the true length
            #gold_id    = gold_id[:sent_len]

            if config.segmenter == 'pointer':
                pointers_predicted = []

                for i in prediction:
                    if i >= sent_len:
                        break
                    pointers_predicted.append(i)

                chunks_predicted = set([(pointers_predicted[i], pointers_predicted[i+1]) for i in xrange(0,len(pointers_predicted) -1, 2)])
                prediction = [2]*sent_len

                for (i, j) in chunks_predicted:
                    prediction[i : j] = [1]*(j-i)
                    prediction[i] = 0

            else:
                prediction = prediction[:sent_len]
                chunks_predicted, _ = get_boundaries(prediction)

            accuracy += np.sum(prediction == gold_id)
            _len     += sent_len

            chunks_gold = set([tuple(gold_id)]) #get_boundaries(gold_id)

            correct_preds += len(chunks_predicted & chunks_gold)
            total_preds += len(chunks_predicted)
            total_correct += len(chunks_gold)
            chunks_predicted_batch.append(chunks_predicted)
            chunks_gold_batch.append(chunks_gold)

        # TODO
        if log_file is not None:
            write_to_file(f, minibatch, predicted_ids,chunks_predicted_batch, chunks_gold_batch)



    if log_file is not None:
        f.close()

    p  = correct_preds / total_preds if correct_preds > 0 else 0
    r  = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return accuracy / float(_len), f1



# Writing to Log file

def write_to_file(f, minibatch, predicted_ids, predicted_chunks, gold_chunks):

    for i in xrange(len(predicted_chunks)):
        curr_sentence = map(lambda idx : inverse_vocab_dict[idx], minibatch.data['context'][i])

        if len(predicted_chunks[i]) > 0:
            predicted_curr_st, predicted_curr_en = list(predicted_chunks[i])[0]
            predicted_curr = curr_sentence[predicted_curr_st: predicted_curr_en+1]
        else:
            predicted_curr = "<NONE>"


        gold_curr_st, gold_curr_en      = list(gold_chunks[i])[0]
        gold_curr      = curr_sentence[gold_curr_st : gold_curr_en+1]

        f.write("%s\n" %(" ".join(curr_sentence)))
        f.write("prediction sequence: %s\n" %(predicted_ids[i]))
        f.write("predictions: %s\n" %(" ".join(predicted_curr)))
        f.write("gold: %s\n" %(" ".join(gold_curr)))


'''
Parse inputs
'''


def get_params():
    parser = argparse.ArgumentParser(description='Mention Segmenter')
    parser.add_argument('-dataset', action="store", default="ACE", dest="dataset", type=str)
    parser.add_argument('-model_name', action="store", default="entity_linker", dest="model_name", type=str)
    parser.add_argument('-segmenter', action="store", default="pointer",dest="segmenter",type=str)

    parser.add_argument('-dropout', action="store", default=0.5, dest="dropout", type=float)
    parser.add_argument('-train', action="store", default=1, dest="train", type=int)
    parser.add_argument('-char_level', action="store", default=0, dest="char_level", type=int)

    parser.add_argument('-base_dir', action="store", default="/iesl/canvas/smurty/epiKB", type=str)
    parser.add_argument('-lr', action="store", default=1e-3, type=float)
    parser.add_argument('-beta1', action="store", default=0.9, type=float)
    parser.add_argument('-beta2', action="store", default=0.999, type=float)
    parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
    parser.add_argument('-weight_decay', action="store", default=0.0, type=float)
    parser.add_argument('-save_model', action="store", default=1, type=int)
    parser.add_argument('-clip_val', action="store", default=2, type=int)
    parser.add_argument('-embedding_dim', action="store", default=300, type=int)
    parser.add_argument('-hidden_dim', action="store", default=150, type=int)
    parser.add_argument('-char_dim', action="store", default=25, type=int)
    parser.add_argument('-char_hidden_dim', action="store", default=25, type=int)

    parser.add_argument('-num_epochs', action="store", default=2, type=int)
    parser.add_argument('-batch_size', action="store", default=256, type=int)
    parser.add_argument('-take_frac', action="store", default=1.0, type=float)

    opts = parser.parse_args(sys.argv[1:])
    for arg in vars(opts):
        print arg, getattr(opts, arg)
    return opts



if __name__ == "__main__":

    opts = get_params()
    run_dir = os.getcwd()
    config_obj = Config_Segmenter(run_dir, opts)

    char_dict = dict( (_char, i+2) for (i, _char) in enumerate([chr(i) for i in range(128)])  )
    char_dict['PAD'] = PAD
    char_dict['OOV'] = OOV
    config_obj.char_size = len(char_dict)

    # === load in all the auxiliary data
    vocab_dict = joblib.load(config_obj.vocab_file)
    config_obj.vocab_size = len(vocab_dict)

    inverse_vocab_dict = {idx: word for (word, idx) in vocab_dict.iteritems()}
    logger.info("\nNumber of words in vocab: %d\n" % config_obj.vocab_size)

    train_data = joblib.load(config_obj.train_file)
    dev_data   = joblib.load(config_obj.dev_file)
    test_data  = joblib.load(config_obj.test_file)

    print("train: %d" %len(train_data))
    print("dev:   %d" %len(dev_data))
    print("test:  %d" %len(test_data))

    pretrained_embeddings = get_trimmed_glove_vectors(config_obj.embedding_file)

    encoder = SentenceRep(config_obj, pretrained_embeddings)
    attribs = ['context', 'char_ids', 'gold_ids']

    if config_obj.segmenter == 'pointer':
        model = PointerNetwork_SoftDot(config_obj, encoder)
        attribs.append('gold_ids_pointer')
    else:
        model = BasicSegmenter(config_obj, encoder)


    load_model(model, config_obj)

    # === Define the training and dev data

    if config_obj.dataset == 'wiki':
        transformer = SegmenterTransformer
    else:
        transformer = SegmenterTransformerAIDA

    train_itr = SegmenterDataset(train_data, attribs, vocab_dict, char_dict, config_obj.batch_size, transformer, True)
    dev_itr   = SegmenterDataset(dev_data, attribs, vocab_dict, char_dict, config_obj.batch_size, transformer, False)
    test_itr  = SegmenterDataset(test_data, attribs, vocab_dict, char_dict, config_obj.batch_size, transformer, False)


    if opts.train:
        logger.info("\n====================TRAINING STARTED====================\n")
        train(model, train_itr, dev_itr, test_itr, config_obj)
    else:
        logger.info("\n====================EVALUATION STARTED====================\n")
        if config_obj.gpu:
            model.cuda()

        _,f1_test = evaluate(model, test_itr, config_obj, "results_test_%s.txt" %config_obj.model_file)
        _,f1_dev = evaluate(model, dev_itr, config_obj,   "results_dev_%s.txt" %config_obj.model_file)

        logger.info("\n: Dev F1 %5.4f" % f1_dev)
        logger.info("\n: Test F1 %5.4f" % f1_test)
