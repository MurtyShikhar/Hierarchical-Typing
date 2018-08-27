import time
import os, sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
from data_iterator import MAX_SENT
# ===== Some utility for padding and creating torch index tensors =====

def _pad_sequences(sequences, pad_tok, max_length, pos):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        if pos:
            seq_ = seq + [idx for idx in xrange(seq[-1]+1, seq[-1] + max_length - len(seq)+1)]
        else:
            seq_ = seq + [pad_tok]*max(max_length - len(seq), 0)


        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return np.array(sequence_padded), np.array(sequence_length)

def pad_sequences(sequences, pad_tok, torchify_op, on_gpu, pos = False):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        Padded and torchified tensors (along with true lengths also torchified) and transferred to GPU
    """
    max_id, max_length_seq = max(enumerate(sequences), key = lambda id : len(id[1]))
    max_length = len(max_length_seq)
    sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length, pos)

    if on_gpu:
        return Variable(torchify_op(sequence_padded).cuda() ), Variable(torchify_op(sequence_length).cuda())
    else:
        return Variable(torchify_op(sequence_padded) ), Variable(torchify_op(sequence_length))

def pad2d(sequences, pad_tok, torchify_op, on_gpu):
    '''

    :param sequences: A list of list of list where the 2nd dimension is for word characters and first dim
                                is for words
    :param pad_tok: everything will be padded with this (usually PAD)
    :param torchify_op:
    :param on_gpu:
    :return: padded, and good for inputting into nn.Module
    '''


    max_length_word = max([max([len(x) for x in seq]) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, pad_tok, max_length_word, False)
        sequence_padded += [sp]
        sequence_length += [sl-1]

    max_length_sentence = max([len(x) for x in sequences])
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word,
                                        max_length_sentence, False)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence, False)

    if on_gpu:
        return Variable(torchify_op(sequence_padded).cuda()), Variable(torchify_op(sequence_length).cuda())
    else:
        return Variable(torchify_op(sequence_padded)), Variable(torchify_op(sequence_length))






# === Functions for loading and saving model ====

def save_model(model, config):
    filename = "%s/%s.checkpoint" %(config.checkpoint_file, config.model_file)

    try:
        torch.save(model.state_dict(), filename)
    except BaseException:
        pass


def load_model(model, config):
    fname = "%s/%s.checkpoint" %(config.checkpoint_file, config.model_file)
    if os.path.isfile(fname):
        model.load_state_dict(torch.load(fname))


# === performing gradient descent

def step(model, loss, optimizer, clip_val):
    optimizer.zero_grad()
    loss.backward()
    # clip gradients
    torch.nn.utils.clip_grad_norm(model.parameters(), clip_val)
    optimizer.step()

# == Helper functions for evaluation (some copied from Riedel EACL 2017)


def get_hierarchical_accuracy(gold_ids, predicted_ids, entity_hierarchy_linked_list):
    '''
        A relaxed evaluation for evaluating concept linking for hierarchically organised concepts
        NOTE: entity_hierarchy_linked_list is a linked list representation of the hierarchy and must not be the transitive closure of hierarchy
    '''
    batch_score = 0.0
    for gold_id, predicted_id in zip(gold_ids, predicted_ids):
        if gold_id == predicted_id:
            batch_score += 1.0
        elif gold_id in entity_hierarchy_linked_list:
            # if predicted ID is a direct parent of the gold ID
            if predicted_id in entity_hierarchy_linked_list[gold_id]:
                batch_score += 1.0
            # if predicted ID is a sibling of the gold ID
            elif predicted_id in entity_hierarchy_linked_list and len(entity_hierarchy_linked_list[predicted_id] & entity_hierarchy_linked_list[gold_id]) > 0:
                batch_score += 1.0

    return batch_score


def f1(p,r):
    if r == 0.:
        return 0.
    return 2 * p * r / float( p + r )


# ============= New class for scoring ===============
class Scorer(object):

    def __init__(self):
        self.acc_accum = 0.0
        self.num_seen = 0

        self.p_macro_accum = 0.0
        self.r_macro_accum = 0.0


        # micro numbers

        self.num_correct_labels = 0.0
        self.num_predicted_labels = 0
        self.num_true_labels = 0


    def get_scores(self):

        p_macro = self.p_macro_accum/self.num_seen
        r_macro = self.r_macro_accum/self.num_seen

        p_micro = self.num_correct_labels/self.num_predicted_labels
        r_micro = self.num_correct_labels/self.num_true_labels

        accuracy = self.acc_accum/self.num_seen

        return accuracy, f1(p_macro, r_macro), f1(p_micro, r_micro)


    def run(self, true_and_prediction):
        self.num_seen += len(true_and_prediction)

        for true_labels, predicted_labels in true_and_prediction:
            self.acc_accum += (set(true_labels) == set(predicted_labels))

            #update micro stats
            self.num_predicted_labels += len(predicted_labels)
            self.num_true_labels += len(true_labels)
            self.num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))

            #update macro stats
            if len(predicted_labels):
                self.p_macro_accum += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            if len(true_labels):
                self.r_macro_accum += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
















def strict(true_and_prediction):
    num_entities = len(true_and_prediction)
    correct_num = 0.
    for true_labels, predicted_labels in true_and_prediction:
        correct_num += set(true_labels) == set(predicted_labels)
    return correct_num

def loose_macro(true_and_prediction):
    num_entities = len(true_and_prediction)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1( precision, recall)

def loose_micro(true_and_prediction):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in true_and_prediction:
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1( precision, recall)


def AP(scores, gold_ids):
    '''
        Calculate the sum of the average precision for this batch
    '''
    aps = []
    for score,gold_id in zip(scores, gold_ids):
        aps.append(average_precision_score(gold_id, score))
    return sum(aps)



def margin_loss_linking(margin):
    '''
    compute the max margin score for entity linking
    scores are negative order violations
    '''

    def f(scores, pos_ids):
        scores_pos = scores.gather(1, pos_ids.unsqueeze(1)) #(batch_size, 1) all positive scores
        return (margin + scores - scores_pos).clamp(min=0).mean()

    return f


def margin_loss_typing(margin):
    '''
    compute the max margin score for typing when there are multiple positives
    '''

    def f(order_violations, pos_ids):
        num_batches = float(order_violations.data.shape[0])
        scores_pos = order_violations*pos_ids
        scores_neg = order_violations - scores_pos
        loss = scores_pos.sum() + (margin - scores_neg).clamp(min=0).sum()
        return loss/num_batches

    return f


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
