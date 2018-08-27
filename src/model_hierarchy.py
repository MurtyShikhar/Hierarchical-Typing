# == Models for pretraining type embeddings to enforce the hierarchy

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal, xavier_uniform
from data_iterator import MAX_SENT
import torch.autograd as autograd
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq


class BilinearHierarchy(nn.Module):
	def __init__(self, config):
		super(BilinearHierarchy, self).__init__()

		self.config = config

		self.type_embeddings = nn.Embedding(self.config.type_size, self.config.embedding_dim)
        self.bilinear_matrix = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias = False)


    def __init__(self, type_structure_data):
    	type_curr_children, type_parent_candidates = type_structure_data
    	type_parent_vecs  = self.type_embeddings(type_parent_candidates) #(batch_size_types , num_total_types, embedding_dim)
    	type_child_vec    = self.bilinear_matrix(self.type_embeddings(type_curr_children)).squeeze().unsqueeze(-1) #(batch_size_types, embedding_dim, 1)
    	type_structure_logits = torch.bmm(type_parent_vecs, type_child_vec).squeeze() #(batch_size_types, num_total_types)

    	return type_structure_logits