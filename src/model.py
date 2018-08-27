import torch
import math
import torch.nn as nn
from torch.nn.init import xavier_normal, xavier_uniform
from data_iterator import MAX_SENT
import torch.autograd as autograd
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq

from itertools import chain

# Some helper functions
def order_violation(parent, child, relu_op):
    # square of the norm of the clipped difference between parent and child
    return torch.pow(relu_op(parent - child), 2).sum(dim=-1)

def log_sum_exp(_tensor):
    '''
       :_tensor is a (batch_size, scores, num_bags) sized torch Tensor
       return another tensor ret of size (batch_size, scores) where ret[i][j] = logsumexp(ret[i][:][j])
    '''
    max_scores, _ = torch.max(_tensor, dim = -1) # (batch_size, scores)
    return max_scores + torch.log(torch.sum(torch.exp(_tensor - max_scores.unsqueeze(-1)), dim = -1)) #(batch_size, scores)


def hermitian_dot(v1, v2):
    '''
        :v1 -> (batch_size, num_candidates, vec_size) and v2 -> (batch_size, vec_size, 1)
    '''

    v1_real, v1_img = v1
    v2_real, v2_img = v2

    return torch.bmm(v1_real, v2_real.unsqueeze(-1)).squeeze(-1) + torch.bmm(v1_img, v2_img.unsqueeze(-1)).squeeze(-1) 


def hermitian_distMult(v1, v2, relation):
    relation_real, relation_img = relation
    v1_real, v1_img = v1
    v2_real, v2_img = v2

    realrealreal =  torch.bmm(v1_real, (v2_real*relation_real).unsqueeze(-1)).squeeze(-1)
    realimgimg   =  torch.bmm(v1_img, (v2_img*relation_real).unsqueeze(-1)).squeeze(-1)
    imgrealimg   =  torch.bmm(v1_img, (v2_real*relation_img).unsqueeze(-1)).squeeze(-1)
    imgimgreal   =  torch.bmm(v1_real, (v2_img*relation_img).unsqueeze(-1)).squeeze(-1)
    return realrealreal + realimgimg + imgrealimg - imgimgreal #(batch_size_struct, num_sampled_parents)


# ====== MENTION ENCODERS ======
# RNNs, CNNs, CNN+postion-embeddings

class MentionEncoderRNN(nn.Module):
    '''
        An object that takes a mention and encodes it. Note
    '''
    # sentence -> (gru -> dropout) x 2 -> affine -> tanh -> concat with (average -> dropout) -> affine -> tanh -> affine -> dropout
    def __init__(self, config, pretrained_weights):
        super(MentionEncoderRNN, self).__init__()

        self.config = config
        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))
        # we do NOT update pretrained embeddings.
        self.word_embeddings.weight.requires_grad = False

        # == currently single directional, but should experiment with bidirectional variants as well, subtracting start - end for bidirectional
        self.gru = nn.GRU(self.config.embedding_dim, self.config.hidden_dim, num_layers = 2, bidirectional = False, dropout = self.config.dropout)

        # gru -> phrase_vec
        self.affine1 = nn.Linear(self.config.hidden_dim, self.config.embedding_dim)
        # [phrase_vec average_vec] -> l1
        self.affine2 = nn.Linear(2*self.config.embedding_dim, self.config.embedding_dim)
        # tanh(l1) -> l2
        self.affine3 = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)


        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)


    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(2, batch_size, self.config.hidden_dim))

        if self.config.gpu:
            return h.cuda()
        else:
            return h


    def forward(self, sentences, mention_representation, start_ids, end_ids):
        # == start_ids = (N, 1), end_ids = (N, 1)

        start_ids = start_ids.repeat(1, self.config.hidden_dim).unsqueeze(1) #(N, 1, hidden_dim)
        end_ids   = end_ids.repeat(1, self.config.hidden_dim).unsqueeze(1) #(N, 1, hidden_dim)

        sentences_embedded = self.word_embeddings(sentences).permute(1, 0, 2) #(W, N, D)
        batch_size = sentences_embedded.data.shape[1]
        hidden     = self.init_hidden(batch_size)
        output, _  = self.gru(sentences_embedded, hidden) #(W, N, hidden_dim)
        output = output.permute(1, 0, 2)
        #print(output.data.shape, start_ids.data.shape)
        phrase_vec = self.tanh(self.affine1(output.gather(1, end_ids).squeeze(1) - output.gather(1, start_ids).squeeze(1))) #(N, D)

        mention_representation = self.dropout(mention_representation) # (N, D)
        context_representation = self.tanh(self.affine2(torch.cat((phrase_vec, mention_representation), dim = -1)))

        return self.dropout(self.affine3(context_representation))



class MentionEncoderCNNWithPosition(nn.Module):
    '''
        Takes a sentence with position embeddings and applies a CNN to it (also uses tanh instead of ReLU)
    '''

    def __init__(self, config, pretrained_weights):
        super(MentionEncoderCNNWithPosition, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))

        self.word_embeddings.weight.requires_grad = False

        self.position_embeddings = nn.Embedding(2*MAX_SENT, self.config.embedding_dim)

        self.conv = nn.Conv1d(self.config.embedding_dim, self.config.embedding_dim, config.kernel_width, stride=1)


        self.affine1 = nn.Linear(2*self.config.embedding_dim, self.config.embedding_dim)
        self.affine3 = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)

        if self.config.features:
            self.feature_embeddings = nn.Embedding(self.feature_size, self.feature_dim)    
            rep_dim = 2*self.config.embedding_dim + self.feature_dim
        else
            rep_dim = 2*self.config.embedding_dim

        self.affine2 = nn.Linear(rep_dim, self.config.embedding_dim)

    def forward(self, sentences, positions, mention_representation, feature_data = None):
        '''
        sentences is a (N, W) matrix with ids
        mention_representation is just a (N, word_dim) matrix with average vectors of the surface form
        '''

        sentences_embedded = self.word_embeddings(sentences) #(N, W, D)
        position_embedded  = self.position_embeddings(positions) #(N, W, D)
        cnn_input_embedding = self.affine1(torch.cat((sentences_embedded, position_embedded), dim = -1) ).transpose(1, 2) #(N, D, W)

        sentence_embedded_conv = self.conv(cnn_input_embedding)
        sentence_embedding, _ = sentence_embedded_conv.max(dim=-1) #(N, D)

        mention_representation = self.dropout(mention_representation)

        if feature_data is not None:
            feature_data_embedding = self.dropout(torch.sum(self.feature_embeddings(feature_data), dim = 1)) #(N, feat_dim)
            context_representation = torch.cat((sentence_embedding, mention_representation, feature_data_embedding), dim = -1) #(N, rep_dim)

        else:
            context_representation = torch.cat((sentence_embedding, mention_representation), dim = -1). #(N, rep_dim)


        context_representation = self.tanh(self.affine2(context_representation))

        return self.dropout(self.affine3(context_representation))


class MentionEncoderCNN(nn.Module):
    '''
        Takes a sentence and applies a CNN to it, and also affines everything
    '''
    def __init__(self, config, pretrained_weights):
        super(MentionEncoderCNN, self).__init__()

        self.config = config
        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))

        self.word_embeddings.weight.requires_grad = False

        self.conv = nn.Conv1d(self.config.embedding_dim, self.config.embedding_dim, config.kernel_width, stride=1)


        self.affine1 = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.affine3 = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)

        if self.config.features:
            self.feature_embeddings = nn.Embedding(self.feature_size, self.feature_dim)    
            rep_dim = 2*self.config.embedding_dim + self.feature_dim
        else
            rep_dim = 2*self.config.embedding_dim

        self.affine2 = nn.Linear(rep_dim, self.config.embedding_dim)




    def forward(self, sentences, mention_representation, feature_data = None):
        '''
            sentences is a (N, W) matrix with ids
            mention_representation is just a (N, word_dim) matrix with average vectors of the surface form
        '''
        sentences_embedded = self.affine1(self.word_embeddings(sentences)).transpose(1,2) #(N, D, W)
        sentence_embedded_conv = self.conv(sentences_embedded) #(N, D, W2)
        sentence_embedding, _ = sentence_embedded_conv.max(dim=-1) #(N, D)

        mention_representation = self.dropout(mention_representation)
        context_representation = torch.cat((sentence_embedding,  mention_representation), dim=-1) #(N, 2*D)


        if feature_data is not None:
            feature_data_embedding = self.dropout(torch.sum(self.feature_embeddings(feature_data), dim = 1)) #(N, feat_dim)
            context_representation = torch.cat((context_representation, feature_data_embedding), dim = -1) #(N, rep_dim)


        context_representation = self.tanh(self.affine2(context_representation))

        return self.dropout(self.affine3(context_representation))



class MentionEncoderBasic(nn.Module):
    '''
        An object that takes a mention and encodes it
    '''

    def __init__(self, config, pretrained_weights):
        super(MentionEncoderBasic, self).__init__()

        self.config = config
        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))
        # we do NOT update pretrained embeddings.
        self.word_embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(p=config.dropout)
        self.affine = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)

    def forward(self, sentences, mention_representation):
        context_representation_affined = self.affine(self.dropout(mention_representation))
        return context_representation_affined


class BasicLinker(nn.Module):

    def __init__(self, config, encoder):
        '''
        :param config: A config object that specifies the hyperparameters of the model
        :param encoder: A Encoder for encoding mentions
        '''

        super(BasicLinker, self).__init__()

        self.config = config
        # == define the encoder that will encode the context modularly
        self.encoder = encoder

        # == define the entity embeddings (Note: these are order embeddings) and so are initialised to be non-negative
        self.entity_embeddings = nn.Embedding(self.config.entity_size, self.config.embedding_dim)

        self.relu = nn.ReLU()


    def forward(self, padded_left_contexts, left_context_lens, padded_right_contexts, right_contexts_lens, entity_candidates):
        '''
        :param mention: A mention object
        :return: unnormalized log probabilities (logits) of gold entity given this mention
        '''

        # == run the mention encoder
        context_representation_affined = self.encoder(padded_left_contexts, left_context_lens, padded_right_contexts, right_contexts_lens).unsqueeze(1) #(batch_size, 1, embedding_dim)
        # get the entity vector
        candidate_ids = self.relu(self.entity_embeddings(entity_candidates))  #(batch_size, num_candidates, embedding_dim) always positive!
        scores = -1.0*(self.relu(candidate_ids - context_representation_affined).norm(2, dim= -1)) #(batch_size, num_candidates)

        #[ : , entity_lens .. entity_candidates.size()[-1]] = -INF
        return scores


class ConstrainedLinker(nn.Module):
    def __init__(self, config, encoder):
        super(ConstrainedLinker, self).__init__()
        self.config = config
        self.encoder = encoder

        # == define the entity embeddings for linking and for typing (Note: these are both order embeddings)
        self.entity_embeddings_union = nn.Embedding(self.config.entity_size, self.config.embedding_dim)
        self.entity_embeddings_intersection = nn.Embedding(self.config.entity_size, self.config.embedding_dim)

        self.relu = nn.ReLU()

    def forward(self, padded_left_contexts, left_context_lens, padded_right_contexts, right_contexts_lens, entity_candidates):
        # == run the mention encoder
        context_representation_affined = self.encoder(padded_left_contexts, left_context_lens, padded_right_contexts, right_contexts_lens).unsqueeze(1) #(batch_size, 1, embedding_dim)

        # get the entity vector
        candidate_ids_union = self.relu(self.entity_embeddings_union(entity_candidates))  #(batch_size, num_candidates, embedding_dim) always positive!
        candidate_ids_intersection = self.relu(self.entity_embeddings_intersection(entity_candidates))  #(batch_size, num_candidates, embedding_dim) always positive!

        scores_1 = -1.0*(self.relu(candidate_ids_union - context_representation_affined).norm(2, dim= -1)) #(batch_size, num_candidates)
        scores_2 = -1.0*(self.relu(context_representation_affined - candidate_ids_intersection).norm(2, dim = -1)) #(batch_size, num_candidates)

        return scores_1 + scores_2



# ==== Linker for UMLS
class Linker_complex(nn.Module):

    def __init__(self, config, encoder):
        '''
        :param config: A config object that specifies the hyperparameters of the model
        :param encoder: A Encoder for encoding mentions
        '''

        super(Linker_complex, self).__init__()

        self.config = config
        # == define the encoder that will encode the context modularly
        self.encoder = encoder

        # == define the entity embeddings
        self.entity_embeddings_real = nn.Embedding(self.config.entity_size, self.config.embedding_dim)
        self.entity_embeddings_img  = nn.Embedding(self.config.entity_size, self.config.embedding_dim)


        # == learnable weight for weighing in the character level tfidf features
        if self.config.priors:
            self.char_feature_weight  = nn.Parameter(torch.FloatTensor([1]))
            self.model_feature_weight = nn.Parameter(torch.FloatTensor([0]))
        else:
            self.char_feature_weight  = 0 
            self.model_feature_weight = nn.Parameter(torch.FloatTensor([1]))

        self.relation_real = nn.Parameter(torch.FloatTensor(self.config.embedding_dim)) 
        self.relation_img  = nn.Parameter(torch.FloatTensor(self.config.embedding_dim))

        # correct initialization
        stdv = 1.0 / math.sqrt(self.config.embedding_dim)
        self.relation_real.data.uniform_(-stdv, stdv)
        self.relation_img.data.uniform_(-stdv, stdv)


        self.encoder_real_mlp = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.encoder_img_mlp  = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, entity_candidates, mention_representation, sentence, char_scores, aux_data, structure_data = None, type_data = None):
        '''
        :return: unnormalized log probabilities (logits) of gold entity given this mention
        '''

        # == run the mention encoder
        if self.config.encoder == "basic":
            context_representation = self.encoder(sentence, mention_representation)
        elif self.config.encoder == "position_cnn":
            context_representation = self.encoder(sentence, aux_data, mention_representation)
        elif self.config.encoder == "rnn_phrase":
            st_ids = aux_data[0]
            en_ids = aux_data[1]
            context_representation = self.encoder(sentence, mention_representation, st_ids, en_ids)

        # get the entity vector
        candidate_ids = [self.entity_embeddings_real(entity_candidates), self.entity_embeddings_img(entity_candidates)]  #(batch_size, num_candidates, embedding_dim)
        context_representation = [self.dropout(self.encoder_real_mlp(context_representation)), self.dropout(self.encoder_img_mlp(context_representation))] #(batch_size, embedding_dim)

        if self.config.learn_graph:
            model_scores = hermitian_distMult(candidate_ids, context_representation, [self.relation_real, self.relation_img])
        else:
            model_scores = hermitian_dot(candidate_ids, context_representation).squeeze() 

        scores = self.model_feature_weight*model_scores + char_scores*self.char_feature_weight  # (batch_size, num_candidates)

        all_scores = {'linking_logits': scores}

        #=== STRUCTURE LOSS
        if structure_data is not None:
            sampled_child_nodes, sampled_parent_nodes = structure_data
            parent_vecs  = [self.entity_embeddings_real(sampled_parent_nodes), self.entity_embeddings_img(sampled_parent_nodes)] #(batch_size_struct , num_sampled_parents, embedding_dim)
            child_vec    = [self.entity_embeddings_real(sampled_child_nodes).squeeze(), self.entity_embeddings_img(sampled_child_nodes).squeeze()]   #(batch_size_struct, embedding_dim)

            if self.config.asymmetric:
                imgrealimg   =  torch.bmm(parent_vecs[1], (child_vec[0]*self.relation_img).unsqueeze(-1)).squeeze()
                imgimgreal   =  torch.bmm(parent_vecs[0], (child_vec[1]*self.relation_img).unsqueeze(-1)).squeeze()
                structure_logits = imgrealimg - imgimgreal #(batch_size_struct, num_sampled_parents)
            else:    
                structure_logits = hermitian_distMult(parent_vecs, child_vec, [self.relation_real, self.relation_img])

            all_scores['structure_logits'] = structure_logits


        return all_scores





# ==== Linker for UMLS
class Linker(nn.Module):

    def __init__(self, config, encoder):
        '''
        :param config: A config object that specifies the hyperparameters of the model
        :param encoder: A Encoder for encoding mentions
        '''

        super(Linker, self).__init__()

        self.config = config
        # == define the encoder that will encode the context modularly
        self.encoder = encoder

        # == define the entity embeddings
        self.entity_embeddings = nn.Embedding(self.config.entity_size, self.config.embedding_dim)

        # == learnable weight for weighing in the character level tfidf features
        if self.config.priors:
            self.char_feature_weight  = nn.Parameter(torch.FloatTensor([1]))
            self.model_feature_weight = nn.Parameter(torch.FloatTensor([0]))
        else:
            self.char_feature_weight  = 0 
            self.model_feature_weight = nn.Parameter(torch.FloatTensor([1]))

       
        self.bilinear_matrix = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias = bool(config.bilinear_bias))

        if self.config.typing_weight > 0:
            self.type_embeddings = nn.Embedding(self.config.type_size, self.config.embedding_dim)



    def forward(self, entity_candidates, mention_representation, sentence, char_scores, aux_data, structure_data = None, typing_data = None):
        '''
        :return: unnormalized log probabilities (logits) of gold entity given this mention
        '''

        # == run the mention encoder
        if self.config.encoder == "basic":
            context_representation = self.encoder(sentence, mention_representation)
        elif self.config.encoder == "position_cnn":
            context_representation = self.encoder(sentence, aux_data, mention_representation)
        elif self.config.encoder == "rnn_phrase":
            st_ids = aux_data[0]
            en_ids = aux_data[1]
            context_representation = self.encoder(sentence, mention_representation, st_ids, en_ids)

        # get the entity vector
        candidate_ids = self.entity_embeddings(entity_candidates)  #(batch_size, num_candidates, embedding_dim)

        if self.config.learn_graph:
            context_representation = self.bilinear_matrix(context_representation).unsqueeze(-1)
            model_scores = torch.bmm(candidate_ids, context_representation)
        else:
            context_representation = context_representation.unsqueeze(-1)
            model_scores = torch.bmm(candidate_ids, context_representation).squeeze() 

        scores = self.model_feature_weight*model_scores + char_scores*self.char_feature_weight  # (batch_size, num_candidates)

        all_scores = {'linking_logits': scores}

        #=== STRUCTURE LOSS
        if structure_data is not None:
            sampled_child_nodes, sampled_parent_nodes = structure_data
            parent_vecs  = self.entity_embeddings(sampled_parent_nodes) #(batch_size_struct , num_sampled_parents, embedding_dim)
            child_vec    = self.bilinear_matrix(self.entity_embeddings(sampled_child_nodes)).squeeze(1).unsqueeze(-1) #(batch_size_struct, embedding_dim, 1)
            structure_logits = torch.bmm(parent_vecs, child_vec).squeeze() #(batch_size_struct, num_sampled_parents)

            all_scores['structure_logits'] = structure_logits

        # == TYPING LOSS
        if typing_data is not None:
            type_embeddings = self.type_embeddings(typing_data) #(batch_size_types, num_types, embeding_dim)
            typing_logits   = torch.bmm(type_embeddings, context_representation).squeeze()
            all_scores['typing_logits'] = typing_logits


        return all_scores


# ==== Linker for UMLS 
class Linker_separate(nn.Module):

    def __init__(self, config, encoder):
        '''
        :param config: A config object that specifies the hyperparameters of the model
        :param encoder: A Encoder for encoding mentions
        '''

        super(Linker_separate, self).__init__()

        self.config = config
        # == define the encoder that will encode the context modularly
        self.encoder = encoder

        # == define the entity embeddings
        self.entity_embeddings = nn.Embedding(self.config.entity_size, self.config.embedding_dim)
        self.entity_embeddings_struct = nn.Embedding(self.config.entity_size, self.config.embedding_dim)

        # == learnable weight for weighing in the character level tfidf features
        if self.config.priors:
            self.char_feature_weight  = nn.Parameter(torch.FloatTensor([1]))
            self.model_feature_weight = nn.Parameter(torch.FloatTensor([0]))
        else:
            self.char_feature_weight  = 0 
            self.model_feature_weight = nn.Parameter(torch.FloatTensor([1]))


       
        self.bilinear_matrix = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias = bool(config.bilinear_bias))

        if self.config.typing_weight > 0:
            self.type_embeddings = nn.Embedding(self.config.type_size, self.config.embedding_dim)



    def forward(self, entity_candidates, mention_representation, sentence, char_scores, aux_data, structure_data = None, typing_data = None):
        '''
        :return: unnormalized log probabilities (logits) of gold entity given this mention
        '''

        # == run the mention encoder
        if self.config.encoder == "basic":
            context_representation = self.encoder(sentence, mention_representation)
        elif self.config.encoder == "position_cnn":
            context_representation = self.encoder(sentence, aux_data, mention_representation)
        elif self.config.encoder == "rnn_phrase":
            st_ids = aux_data[0]
            en_ids = aux_data[1]
            context_representation = self.encoder(sentence, mention_representation, st_ids, en_ids)

        # get the entity vector
        candidate_ids = self.entity_embeddings(entity_candidates)  #(batch_size, num_candidates, embedding_dim)
        candidate_ids_struct = self.entity_embeddings_struct(entity_candidates)  #(batch_size, num_candidates, embedding_dim)


        if self.config.learn_graph:
            context_representation = self.bilinear_matrix(context_representation).unsqueeze(-1)
            model_scores = torch.bmm(candidate_ids, context_representation) + torch.bmm(candidate_ids_struct, context_representation)
        else:
            context_representation = context_representation.unsqueeze(-1)
            model_scores = torch.bmm(candidate_ids, context_representation).squeeze() + torch.bmm(candidate_ids_struct, context_representation).squeeze() 

        scores = self.model_feature_weight*model_scores + char_scores*self.char_feature_weight  # (batch_size, num_candidates)

        all_scores = {'linking_logits': scores}

        #=== STRUCTURE LOSS
        if structure_data is not None:
            sampled_child_nodes, sampled_parent_nodes = structure_data
            parent_vecs  = self.entity_embeddings_struct(sampled_parent_nodes) #(batch_size_struct , num_sampled_parents, embedding_dim)
            child_vec    = self.bilinear_matrix(self.entity_embeddings_struct(sampled_child_nodes)).squeeze().unsqueeze(-1) #(batch_size_struct, embedding_dim, 1)
            structure_logits = torch.bmm(parent_vecs, child_vec).squeeze() #(batch_size_struct, num_sampled_parents)

            all_scores['structure_logits'] = structure_logits

        # == TYPING LOSS
        if typing_data is not None:
            type_embeddings = self.type_embeddings(typing_data) #(batch_size_types, num_types, embeding_dim)
            typing_logits   = torch.bmm(type_embeddings, context_representation).squeeze()
            all_scores['typing_logits'] = typing_logits


        return all_scores









'''
    MIL approach to typing entities
'''
class MultiInstanceTyper(nn.Module):
    def __init__(self, config, encoder):
        super(MultiInstanceTyper, self).__init__()
        self.config = config
        self.encoder = encoder

        # later look at methods for composing the type vectors as well
        self.type_embeddings = nn.Embedding(self.config.type_size, self.config.embedding_dim)

        # === Bilinear classifier for learning the structure
        if self.config.struct_weight > 0:
            self.bilinear_matrix = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias = False)

        # === Entity linking parameters for multi-tasking entity linking with typing
        if self.config.linker_weight > 0:
            self.entity_embeddings = nn.Embedding(self.config.entity_size, self.config.embedding_dim)
            self.prior_feature_weight  = nn.Parameter(torch.FloatTensor([1]))
            self.model_feature_weight = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, type_candidates, mention_representation_bag, sentence_bag, aux_data, bag_size, type_structure_data = None, linking_data = None, type_structure_data_entity = None):
        # == run the mention encoder

        num_types = self.config.fb_type_size # predictions are made only on freebase types


        # sentence_bag = (batch_size, bag_size, sentence)
        # mention_representation_bag = (batch_size, bag_size, D)


        if self.config.encoder == "basic":
            context_representation_bag = self.encoder(sentence_bag, mention_representation_bag).unsqueeze(-1)
        elif self.config.encoder == "position_cnn":
            context_representation_bag = self.encoder(sentence_bag, aux_data, mention_representation_bag).unsqueeze(-1)
        elif self.config.encoder == "rnn_phrase":
            st_ids = aux_data[0]
            en_ids = aux_data[1]
            context_representation_bag = self.encoder(sentence_bag, mention_representation_bag, st_ids, en_ids).unsqueeze(-1)

        #(batch_size*bag_size, embedding_dim, 1)

        # get the type vectors
        type_vectors = self.type_embeddings(type_candidates)  #(batch_size*bag_size, num_types, embedding_dim)
        type_context_scores = torch.bmm(type_vectors, context_representation_bag).squeeze(-1).view(-1, bag_size, num_types) # (batch_size* bag_size, num_types)

        # logsumexp along the bag_size axis

        type_context_scores_pooled = log_sum_exp(type_context_scores.transpose(1,2)) #(batch_size, num_types)

        logits = {'typing_logits' : type_context_scores_pooled}

        if type_structure_data is not None:
            type_curr_children, type_parent_candidates = type_structure_data
            type_parent_vecs  = self.type_embeddings(type_parent_candidates) #(batch_size_types , num_total_types, embedding_dim)
            type_child_vec    = self.bilinear_matrix(self.type_embeddings(type_curr_children)).squeeze(1).unsqueeze(-1) #(batch_size_types, embedding_dim, 1)
            type_structure_logits = torch.bmm(type_parent_vecs, type_child_vec).squeeze(-1) #(batch_size_types, num_total_types)

            logits['type_structure_logits'] = type_structure_logits


        if linking_data is not None:
            entity_candidates, entity_candidate_priors = linking_data
            candidate_ids = self.entity_embeddings(entity_candidates)  #(batch_size*bag_size, num_candidates, embedding_dim)
            scores = self.model_feature_weight*torch.bmm(candidate_ids, context_representation_bag).squeeze(-1) + entity_candidate_priors*self.prior_feature_weight  # (batch_size*bag_size, num_candidates)
            logits['linking_logits'] = scores

        if type_structure_data_entity is not None:
            entity_curr_children, type_parent_candidates = type_structure_data_entity
            type_parent_vecs = self.type_embeddings(type_parent_candidates)
            entity_children_vecs = self.bilinear_matrix(self.entity_embeddings(entity_curr_children)).squeeze().unsqueeze(-1) #(batch_size_entities, embedding_dim, 1)
            entity_structure_logits = torch.bmm(type_parent_vecs, entity_children_vecs).squeeze() #(batch_size_entities, num_parent_types)
            logits['entity_structure_logits'] = entity_structure_logits

        return logits


'''
    MIL approach to typing entities
'''
class MultiInstanceTyper_complex(nn.Module):
    def __init__(self, config, encoder):
        super(MultiInstanceTyper_complex, self).__init__()
        self.config = config
        self.encoder = encoder

        # later look at methods for composing the type vectors as well
        self.type_embeddings_real = nn.Embedding(self.config.type_size, self.config.embedding_dim)
        self.type_embeddings_img  = nn.Embedding(self.config.type_size, self.config.embedding_dim)

        self.relation_real = nn.Parameter(torch.FloatTensor(self.config.embedding_dim)) 
        self.relation_img  = nn.Parameter(torch.FloatTensor(self.config.embedding_dim))

        # correct initialization
        stdv = 1.0 / math.sqrt(self.config.embedding_dim)
        self.relation_real.data.uniform_(-stdv, stdv)
        self.relation_img.data.uniform_(-stdv, stdv)


        self.encoder_real_mlp = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.encoder_img_mlp  = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, type_candidates, mention_representation_bag, sentence_bag, aux_data, bag_size, type_structure_data = None, linking_data = None, type_structure_data_entity = None):
        # == run the mention encoder

        num_types = self.config.fb_type_size # predictions are made only on freebase types

        if self.config.encoder == "basic":
            context_representation_bag = self.encoder(sentence_bag, mention_representation_bag)
        elif self.config.encoder == "position_cnn":
            context_representation_bag = self.encoder(sentence_bag, aux_data, mention_representation_bag)
        elif self.config.encoder == "rnn_phrase":
            st_ids = aux_data[0]
            en_ids = aux_data[1]
            context_representation_bag = self.encoder(sentence_bag, mention_representation_bag, st_ids, en_ids)

        # get the type vectors
        context_representation_bag = [self.dropout(self.encoder_real_mlp(context_representation_bag)), self.dropout(self.encoder_img_mlp(context_representation_bag))] #(batch_size, embedding_dim)
        type_vectors = [self.type_embeddings_real(type_candidates), self.type_embeddings_img(type_candidates)]  #(batch_size*bag_size, num_types, embedding_dim)
        type_context_scores = hermitian_dot(type_vectors, context_representation_bag).squeeze().view(-1, bag_size, num_types) 

        # logsumexp along the bag_size axis

        type_context_scores_pooled = log_sum_exp(type_context_scores.transpose(1,2)) #(batch_size, num_types)

        logits = {'typing_logits' : type_context_scores_pooled}

        if type_structure_data is not None:
            type_curr_children, type_parent_candidates = type_structure_data
            parent_vecs  = [self.type_embeddings_real(type_parent_candidates), self.type_embeddings_img(type_parent_candidates)] #(batch_size_types , num_total_types, embedding_dim)
            child_vec    = [self.type_embeddings_real(type_curr_children).squeeze(1), self.type_embeddings_img(type_curr_children).squeeze(1)]   #(batch_size_types, embedding_dim)
            if self.config.asymmetric:
                imgrealimg   =  torch.bmm(parent_vecs[1], (child_vec[0]*self.relation_img).unsqueeze(-1)).squeeze(-1)
                imgimgreal   =  torch.bmm(parent_vecs[0], (child_vec[1]*self.relation_img).unsqueeze(-1)).squeeze(-1)
                structure_logits = imgrealimg - imgimgreal #(batch_size_struct, num_sampled_parents)
            else:    
                structure_logits = hermitian_distMult(parent_vecs, child_vec, [self.relation_real, self.relation_img])


            logits['type_structure_logits'] = structure_logits


        if linking_data is not None:
            entity_candidates, entity_candidate_priors = linking_data
            candidate_ids = self.entity_embeddings(entity_candidates)  #(batch_size*bag_size, num_candidates, embedding_dim)
            scores = self.model_feature_weight*torch.bmm(candidate_ids, context_representation_bag).squeeze(-1) + entity_candidate_priors*self.prior_feature_weight  # (batch_size*bag_size, num_candidates)
            logits['linking_logits'] = scores

        if type_structure_data_entity is not None:
            entity_curr_children, type_parent_candidates = type_structure_data_entity
            type_parent_vecs = self.type_embeddings(type_parent_candidates)
            entity_children_vecs = self.bilinear_matrix(self.entity_embeddings(entity_curr_children)).squeeze().unsqueeze(-1) #(batch_size_entities, embedding_dim, 1)
            entity_structure_logits = torch.bmm(type_parent_vecs, entity_children_vecs).squeeze() #(batch_size_entities, num_parent_types)
            logits['entity_structure_logits'] = entity_structure_logits

        return logits

class BasicMentionTyperDot(nn.Module):
    def __init__(self, config, encoder):
        super(BasicMentionTyperDot, self).__init__()
        self.config = config
        self.encoder = encoder
        # == define the type embeddings
        self.type_embeddings = nn.Embedding(self.config.type_size, self.config.embedding_dim)

        self.bilinear_transform = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias = False)




    def forward(self, type_candidates, mention_representation, sentence, aux_data, feature_data = None, type_structure_data = None):
        # == run the mention encoder
        if self.config.encoder == "basic":
            context_representation = self.encoder(sentence, mention_representation, feature_data).unsqueeze(-1)
        elif self.config.encoder == "position_cnn":
            context_representation = self.encoder(sentence, aux_data, mention_representation, feature_data).unsqueeze(-1)
        elif self.config.encoder == "rnn_phrase":
            st_ids = aux_data[0]
            en_ids = aux_data[1]
            context_representation = self.encoder(sentence, mention_representation, st_ids, en_ids).unsqueeze(-1)



        # get the type vectors
        type_vectors = self.type_embeddings(type_candidates) #(batch_size, num_types, embedding_dim)

        num_types = type_vectors.size()[1]
        type_mention_logits = torch.bmm(type_vectors, context_representation).squeeze() #(batch_size, num_types)

        if type_structure_data is not None:
            type_curr_children, type_parent_candidates = type_structure_data

            type_parent_vecs = self.type_embeddings(type_parent_candidates) #(batch_size2 , num_types, embedding_dim)
            type_child_vec = self.bilinear_transform(self.type_embeddings(type_curr_children)).squeeze().unsqueeze(-1) #(batch_size2, embedding_dim, 1)
            type_structure_logits = torch.bmm(type_parent_vecs, type_child_vec).squeeze()

            return type_mention_logits, type_structure_logits

        else:
            return type_mention_logits

class ComplexMentionTyperDot(nn.Module):
    def __init__(self, config, encoder):
        super(ComplexMentionTyperDot, self).__init__()
        self.config = config
        self.encoder = encoder
        # == define the type embeddings
        self.type_embeddings_real = nn.Embedding(self.config.type_size, self.config.embedding_dim)
        self.type_embeddings_img  = nn.Embedding(self.config.type_size, self.config.embedding_dim)


        self.relation_real = nn.Parameter(torch.FloatTensor(self.config.embedding_dim)) 
        self.relation_img  = nn.Parameter(torch.FloatTensor(self.config.embedding_dim))

        # correct initialization
        stdv = 1.0 / math.sqrt(self.config.embedding_dim)
        self.relation_real.data.uniform_(-stdv, stdv)
        self.relation_img.data.uniform_(-stdv, stdv)

        self.encoder_real_mlp = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.encoder_img_mlp  = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)




    def forward(self, type_candidates, mention_representation, sentence, aux_data, feature_data = None, type_structure_data = None):
        # == run the mention encoder
        if self.config.encoder == "basic":
            context_representation = self.encoder(sentence, mention_representation, feature_data)
        elif self.config.encoder == "position_cnn":
            context_representation = self.encoder(sentence, aux_data, mention_representation, feature_data)
        elif self.config.encoder == "rnn_phrase":
            st_ids = aux_data[0]
            en_ids = aux_data[1]
            context_representation = self.encoder(sentence, mention_representation, st_ids, en_ids)


        context_representation = [self.dropout(self.encoder_real_mlp(context_representation)), self.dropout(self.encoder_img_mlp(context_representation))] #(batch_size, embedding_dim)

        # get the type vectors
        type_vectors = [self.type_embeddings_real(type_candidates), self.type_embeddings_real(type_candidates)] #(batch_size, 1, embedding_dim)


        type_mention_logits = hermitian_dot(type_vectors, context_representation).squeeze() 


        if type_structure_data is not None:

            sampled_child_nodes, sampled_parent_nodes = type_structure_data
            parent_vecs  = [self.type_embeddings_real(sampled_parent_nodes), self.type_embeddings_img(sampled_parent_nodes)] #(batch_size_struct , num_sampled_parents, embedding_dim)
            child_vec    = [self.type_embeddings_real(sampled_child_nodes).squeeze(), self.type_embeddings_img(sampled_child_nodes).squeeze()]   #(batch_size_struct, embedding_dim)

            if self.config.asymmetric:
                imgrealimg   =  torch.bmm(parent_vecs[1], (child_vec[0]*self.relation_img).unsqueeze(-1)).squeeze()
                imgimgreal   =  torch.bmm(parent_vecs[0], (child_vec[1]*self.relation_img).unsqueeze(-1)).squeeze()
                type_structure_logits = imgrealimg - imgimgreal #(batch_size_struct, num_sampled_parents)
            else:    
                type_structure_logits = hermitian_distMult(parent_vecs, child_vec, [self.relation_real, self.relation_img])

            return type_mention_logits, type_structure_logits

        else:
            return type_mention_logits


# == Order embeddings are optimized only via max margin
class BasicMentionTyperOrder(nn.Module):
    def __init__(self, config, encoder):
        super(BasicMentionTyperOrder, self).__init__()
        self.config = config
        self.encoder = encoder

        # == define the type embeddings (these are order embeddings)
        self.type_embeddings = nn.Embedding(self.config.type_size, self.config.embedding_dim)

        self.relu = nn.ReLU()


    def forward(self, type_candidates, mention_representation, sentence = None, type_structure_data = None):
        # == run the mention encoder
        context_representation_affined = self.relu(self.encoder(sentence, mention_representation)).unsqueeze(1) #(batch_size, 1, embedding_dim)

        # get the type vectors
        type_vectors = self.relu(self.type_embeddings(type_candidates)) #(batch_size, num_types, embedding_dim)

        # type mention factor order violations
        type_mention_ov = order_violation(type_vectors, context_representation_affined, self.relu) #(batch_size, num_types)

        prob_type_given_mention = torch.exp(context_representation_affined.norm(1, dim = -1) - torch.max(context_representation_affined, type_vectors).norm(1, dim = -1))

        if type_structure_data is not None:
            type_curr_children, type_parent_candidates = type_structure_data
            type_parent_vecs = self.relu(self.type_embeddings(type_parent_candidates)) #(batch_size2 , num_types, embedding_dim)
            type_child_vec = self.relu(self.type_embeddings(type_curr_children)) #(batch_size2, 1, embedding_dim)
            type_structure_ov = order_violation(type_parent_vecs, type_child_vec, self.relu) #(batch_size2, num_types)

            return -1.0*type_mention_ov, -1.0*type_structure_ov, prob_type_given_mention

        return -1.0*type_mention_ov, prob_type_given_mention
