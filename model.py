import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from src.spodernet.spodernet.utils.global_config import Config
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from torch.nn.init import xavier_normal_, xavier_uniform_
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.init as init
import os, sys
path_dir = os.getcwd()


timer = CUDATimer()
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred

class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.weight = Parameter(torch.Tensor(in_features, out_features).uniform_(-1, 1))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        #self.weight = Parameter(torch.FloatTensor(num_relations, in_features, out_features))
        #self.weight_sum = Parameter(torch.Tensor(num_relations).uniform_(-1, 1))
        self.weight_sum = Parameter(torch.FloatTensor(num_relations))


        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_sum.size(0))
        self.weight_sum.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):

        for i in range(len(adj)):
            if i == 0:
                output = adj[i].mul(self.weight_sum[i].tolist())
            else:
                output.add_(adj[i].mul(self.weight_sum[i].tolist()))
        support = torch.mm(input, self.weight)
        output = torch.spmm(output, support)


        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


'''
class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)

        self.gc1 = GraphConvolution(Config.embedding_dim, 100, num_relations)
        self.gc2 = GraphConvolution(100, Config.embedding_dim, num_relations)


        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368,Config.embedding_dim)

        self.bn3 = torch.nn.BatchNorm1d(100)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)


        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)
        #self.gc1.init()
        #self.gc1.init()

    def forward(self, e1, rel, X, A):

        emb_initial = self.emb_e(X)

        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        #x = torch.nn.functional.normalize(self.gc1(emb_initial, A))
        x = F.tanh(x)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.bn4(self.gc2(x, A))
        #x=self.gc2(x, A)
        e1_embedded_all = F.tanh(x)
        #x = F.relu(self.gc1(emb_initial, A))
        #x = F.dropout(x, 0.5, training=self.training)
        #e1_embedded_all = F.relu(self.gc2(x, A))
        e1_embedded = e1_embedded_all[e1].view(-1, 1, 10, 20)

        #e1_embedded = self.emb_e(e1).view(-1, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        #print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred
'''

class SACN(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(SACN, self).__init__()

        init_emb_size = 200
        gc1_emb_size = 100
        self.emb_e = torch.nn.Embedding(num_entities, init_emb_size, padding_idx=0)

        self.gc1 = GraphConvolution(init_emb_size, gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(gc1_emb_size, Config.embedding_dim, num_relations)


        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        #self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.hidden_drop = torch.nn.Dropout(0.2)
        #self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.feature_map_drop = torch.nn.Dropout(0.2)
        self.loss = torch.nn.BCELoss()

        self.conv1 =  nn.Conv1d(2, 100, 1, stride=1)
        #self.conv2 = torch.nn.Conv1d(1, 50, 3, stride=2)
        #self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        #self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn0 = torch.nn.BatchNorm1d(2)
        #self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(20000,Config.embedding_dim)

        self.bn3 = torch.nn.BatchNorm1d(gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(init_emb_size)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)
        #self.gc1.init()
        #self.gc1.init()

    def forward(self, e1, rel, X, A):

        emb_initial = self.emb_e(X)
        #emb_initial = self.bn_init(emb_initial)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        #x = torch.nn.functional.normalize(self.gc1(emb_initial, A))
        x = F.tanh(x)
        x = F.dropout(x, 0.2, training=self.training)

        x = self.bn4(self.gc2(x, A))
        #x=self.gc2(x, A)
        e1_embedded_all = F.tanh(x)
        #x = F.relu(self.gc1(emb_initial, A))
        e1_embedded_all = F.dropout(e1_embedded_all, 0.2, training=self.training)
        #e1_embedded_all = F.relu(self.gc2(x, A))
        #e1_embedded = e1_embedded_all[e1].view(-1, 1, 10, 20)
        e1_embedded = e1_embedded_all[e1]

        #e1_embedded = self.emb_e(e1).view(-1, 1, 10, 20)
        #rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)
        rel_embedded = self.emb_rel(rel)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        #print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        #x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred
