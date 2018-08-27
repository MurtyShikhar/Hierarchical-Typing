# == Models for pretraining type embeddings to enforce the hierarchy

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal, xavier_uniform
from data_iterator import MAX_SENT
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq


# ========= ENCODER MODEL ==========
class SentenceRep(nn.Module):
    '''
        Returns a sentence representation
    '''

    def __init__(self, config, pretrained_weights):
        super(SentenceRep, self).__init__()

        self.config = config

        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))
        # we do NOT update pretrained embeddings.
        self.word_embeddings.weight.requires_grad = False


        # add character level LSTM
        if self.config.char_level:
            self.char_embeds = nn.Embedding(self.config.char_size, self.config.char_dim)
            self.char_gru = nn.GRU(self.config.char_dim, self.config.char_hidden_dim, num_layers=1, bias = True, batch_first = True,
                                   dropout=0.0, bidirectional=True)

            self.embedding_dim = self.config.embedding_dim + 2*self.config.char_hidden_dim

        else:
            self.embedding_dim = self.config.embedding_dim


        # encode the input representation
        self.gru = nn.GRU(self.embedding_dim, self.config.hidden_dim, num_layers=1, bias = True, batch_first = True, dropout = 0.0, bidirectional=True)
        # 2.0*hidden_dim -> hidden_dim
        self.affine1 = nn.Linear(2*self.config.hidden_dim, self.config.hidden_dim, bias = False)

        #dropout
        self.dropout = nn.Dropout(self.config.dropout)

    def init_hidden(self, batch_size, dim):
        h = Variable(torch.zeros(2, batch_size, dim))

        if self.config.gpu:
            return h.cuda()
        else:
            return h


    def forward(self, sentence, char_ids, char_lens):
        '''

        :param sentence: (N, W) sentence IDs padded with PAD_TOK
        :param char_ids: (N, W, W') char IDs padded with PAD_TOK
        :param char_lens: (N, W) length of every word padded with 0s
        :return: (N, W, 2*hidden_dim)
        '''


        sentence_embedded = self.word_embeddings(sentence) # (N, W, D)
        if self.config.char_level:
            batch_size, sent_size, char_len = char_ids.size()
            char_ids = char_ids.view(-1, char_len)
            char_embeddings = self.char_embeds(char_ids) #(N*W, W', char_dim)

            hidden_char = self.init_hidden(batch_size*sent_size, self.config.char_hidden_dim)
            char_embeddings_gru, _ = self.char_gru(char_embeddings, hidden_char) #(N*W, W', 2*char_hidden_dim)
            char_embeddings_fwd_gru  = char_embeddings_gru[:, :, :self.config.char_hidden_dim]
            char_embeddings_back_gru = char_embeddings_gru[:, 0, self.config.char_hidden_dim:].contiguous().view(batch_size, -1,
                                                                                                    self.config.char_hidden_dim)
            char_embeddings_fwd_gru  = torch.gather(char_embeddings_fwd_gru, 1,
                                                    char_lens.view(-1, 1, 1).repeat(1, 1, self.config.char_hidden_dim)).contiguous().view(batch_size,
                                                                                                                             -1, self.config.char_hidden_dim)
            sentence_embedded = self.dropout(torch.cat((sentence_embedded, char_embeddings_fwd_gru, char_embeddings_back_gru), dim=2))

        else:
            batch_size = sentence_embedded.size()[0]
            sentence_embedded = self.dropout(sentence_embedded)


        hidden = self.init_hidden(batch_size, self.config.hidden_dim)
        gru_out, h =  self.gru(sentence_embedded, hidden) # ( N, W, 2*hidden_dim)

        return self.affine1(gru_out.contiguous()), self.affine1(torch.cat([h[0], h[1]], dim=-1))




# ======= BASIC DECODER =======
class BasicSegmenter(nn.Module):
    '''
    Sequence Labelling with BIO tags. Also has characters at times
    '''

    def __init__(self, config, encoder):
        super(BasicSegmenter, self).__init__()

        self.config = config
        self.encoder = encoder

        if self.config.char_level:
            self.embedding_dim = self.config.embedding_dim + 2*self.config.char_hidden_dim
        else:
            self.embedding_dim = self.config.embedding_dim

        self.affine1 = nn.Linear(2*self.config.hidden_dim, self.embedding_dim)
        self.gru2output = nn.Linear(self.embedding_dim, 3) # B,I,O

        #dropout
        self.dropout = nn.Dropout(self.config.dropout)

        # non-linearities
        self.tanh = nn.Tanh()


    def forward(self, sentence, sentence_lens, char_ids, char_lens, pointer_answers):
        '''

        :param sentence: (N, W) sentence IDs padded with PAD_TOK
        :param char_ids: (N, W, W') char IDs padded with PAD_TOK
        :param char_lens: (N, W) length of every word padded with 0s
        :return: logits
        '''

        gru_out = self.encoder(sentence, char_ids, char_lens)
        bio_features = self.dropout(self.tanh(self.affine1(gru_out.view(-1, 2*self.config.hidden_dim))))
       

        logits = self.gru2output(bio_features)
        return logits #(-1, 3) first W contains first sentence and so on



# ======= POINTER NET v0.1 =======
class PointerNetwork(nn.Module):
    '''
    Can detect mention boundaries given a sentence.
    '''

    def __init__(self, config, encoder):
        super(PointerNetwork, self).__init__()

        self.config = config
        self.encoder = encoder

        # encode the input representation
        self.gru_cell = nn.GRUCell(2*self.config.hidden_dim, self.config.hidden_dim)

        # parameters for attention
        self.V = nn.Linear(2*self.config.hidden_dim, self.config.hidden_dim, bias=False)
        self.W = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)
        #bias initalise to 0
        #self.W.bias.data.fill_(0)

        self.v = nn.Parameter(torch.Tensor(1, self.config.hidden_dim, 1))
        # initialise the above correctly
        stdv = 1.0 / math.sqrt(self.config.hidden_dim)
        self.v.data.uniform_(-stdv, stdv)


        # non linearity
        self.tanh = nn.Tanh()
        self.softmax_op = nn.Softmax()

    def init_hidden(self, batch_size, dim):
        h = Variable(torch.zeros(batch_size, dim))

        if self.config.gpu:
            return h.cuda()
        else:
            return h


    def forward(self, sentence, sentence_lens, char_ids, char_lens, pointer_answers):
        '''

        :param sentence: (N, W)
        :param sentence_lens: (N,)
        :param char_ids: (N, W, W')
        :param char_lens: (N, W)
        :param pointer_answers:  (N, k)
        :return: (N*k, W+1) logits
        '''
        steps = range(pointer_answers.size(1))
        batch_size, sent_size = sentence.size()


        context = self.encoder(sentence, char_ids, char_lens).permute(1, 0, 2) #(W, N, 2*H)



        zeros = Variable(torch.zeros(1, batch_size, 2*self.config.hidden_dim)).cuda() if self.config.gpu else Variable(torch.zeros(1, batch_size, 2*self.config.hidden_dim)) 

        context_new = torch.cat((context, zeros), dim = 0) #(W+1, N, 2*H)

        # mask things beyond actual length with 0
        for i in xrange(sent_size):
            context_new[i] = (1 - (i >= sentence_lens+1).float()*1e10)*context[i]

        context_transformed = self.V(context_new)  #(W+1, N, H)
        hidden = self.init_hidden(batch_size, self.config.hidden_dim) #(N, H)
        output = []


        for i in steps:
            f_k = self.tanh(context_transformed + self.W(hidden).repeat(sent_size+1, 1, 1)) #(W+1, N, H)
            f_k_dotted  = torch.bmm(f_k, self.v.repeat(sent_size+1, 1, 1)).squeeze(-1)  #(W+1, N)
 

            beta_k = F.softmax(f_k_dotted.transpose(1,0)).transpose(1,0) #(W+1, N)

            output.append(beta_k.transpose(1,0)) #(N, W+1)
            context_aligned = torch.sum(context_new*(beta_k.unsqueeze(-1).repeat(1, 1, 2*self.config.hidden_dim)), dim = 0)  #(N, 2*H)
            hidden = self.gru_cell(context_aligned, hidden)


        # output is k*N, W+1
        output = torch.cat(output, 0).view(-1, batch_size, sent_size+1).permute(1,0,2).contiguous().view(-1, sent_size+1)
        return output














# ======= POINTER NET v0.2 =======
class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: N x dim
        context: N x W+1 x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # N x dim x 1

        # Get attention
        _attn = torch.bmm(context, target).squeeze(2)  # N x W+1
        attn = self.sm(_attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # N x 1 x W+1

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # N x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, _attn


class PointerNetwork_SoftDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, config, encoder):
        """Initialize params."""
        super(PointerNetwork_SoftDot, self).__init__()
        self.config  = config
        self.encoder = encoder 

        self.input_size = self.config.hidden_dim
        self.hidden_size = self.config.hidden_dim


        self.input_weights = nn.Linear(self.input_size, 4 * self.hidden_size)
        self.hidden_weights = nn.Linear(self.hidden_size, 4 * self.hidden_size)

        self.attention_layer = SoftDotAttention(self.hidden_size)

        self.stop_vector = nn.Parameter(torch.randn(1, 1, self.input_size))
        # initialise the above correctly
        stdv = 1.0 / math.sqrt(self.input_size)
        self.stop_vector.data.uniform_(-stdv, stdv)

    def init_hidden(self, batch_size, dim):
        h = Variable(torch.zeros(batch_size, dim))

        if self.config.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, sentence, sentence_lens, char_ids, char_lens, pointer_answers):
        '''
        :param sentence: (N, W)
        :param sentence_lens: (N,)
        :param char_ids: (N, W, W')
        :param char_lens: (N, W)
        :param pointer_answers:  (N, k)
        '''

        def recurrence(input, hidden, ctx):
            """Recurrence helper."""
            hx, cx = hidden  # N x hidden_size
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # N x hidden_size
            h_tilde, alphas = self.attention_layer(hy, ctx.transpose(0, 1))

            return (h_tilde, cy), alphas



        batch_size, sent_size = sentence.size()
        pointer_answers = pointer_answers.transpose(1, 0).contiguous() # (k , N)

        context, h = self.encoder(sentence, char_ids, char_lens) 
        context = context.permute(1, 0, 2) #(W, N, input_size)
        context_new = torch.cat((context, self.stop_vector.repeat(1, batch_size, 1)), dim = 0) #(W+1, N, input_size)
        hidden = h, h #(N, hidden_size)


        # now run the pointer network
        output = []
        steps = range(pointer_answers.size(0))

        for i in steps:
            if i > 1:
                _, pointer_idx = output[-1].max(dim=-1)
                curr_inputs = context_new.permute(1,0,2).gather(1, pointer_idx.view(batch_size, 1, 1).repeat(1, 1, self.input_size)).squeeze(1) #(N, input_size)
            else:
                curr_inputs = self.init_hidden(batch_size, self.input_size) #(N, input_size)

            hidden, alphas = recurrence(curr_inputs, hidden, context_new)
            output.append(alphas) #(N, W+1)


        # output is k*N, W+1
        output = torch.cat(output, 0).view(-1, batch_size, sent_size+1).permute(1,0,2).contiguous().view(-1, sent_size+1)
        return output
