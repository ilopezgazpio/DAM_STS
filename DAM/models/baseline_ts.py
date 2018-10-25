'''
baseline model for Stanford natural language inference
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class encoder(nn.Module):

    def __init__(self, num_embeddings, embedding_size, hidden_size, bigrams = False, trigrams = False):
        super(encoder, self).__init__()

        self.bigrams = bigrams
        self.trigrams = trigrams

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.input_linear = nn.Linear(self.embedding_size, self.hidden_size, bias=False)

        if self.bigrams:
            self.relu1 = nn.ReLU()
            self.conv1 = nn.Conv1d(self.embedding_size, self.hidden_size, 2,  padding=1, bias=False)

        if self.trigrams:
            self.relu2 = nn.ReLU()
            self.conv2 = nn.Conv1d(self.embedding_size, self.hidden_size, 3,  padding=1, bias=False)

        ''' weight init'''
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                init.xavier_uniform(m.weight)

    def forward(self, sent1, sent2):
        '''
               sent: batch_size x length (Long tensor)
        '''
        batch_size = sent1.size(0)
        sent1 = self.embedding(sent1)
        sent2 = self.embedding(sent2)
        # sent: batch x words x words_emb
        
        # to call inpu_linear we change the size of sent to (-1,embedding_size), and then back
        sent1_linear = self.input_linear(sent1.view(-1, self.embedding_size)).view(batch_size, -1, self.hidden_size)
        sent2_linear = self.input_linear(sent2.view(-1, self.embedding_size)).view(batch_size, -1, self.hidden_size)
        
        if (self.bigrams):
            # to call conv we change to <batch x input_size (embs) x seq_len(words)> and back
            bigrams1 = self.relu1(self.conv1(sent1.transpose(1,2)).transpose(1,2))
            bigrams2 = self.relu1(self.conv1(sent2.transpose(1,2)).transpose(1,2))

            sent1_linear = torch.cat((sent1_linear, bigrams1), 1)
            sent2_linear = torch.cat((sent2_linear, bigrams2), 1)

        if (self.trigrams):
            trigrams1 = self.relu2(self.conv2(sent1.transpose(1,2)).transpose(1,2))
            trigrams2 = self.relu2(self.conv2(sent2.transpose(1,2)).transpose(1,2))

            sent1_linear = torch.cat((sent1_linear, trigrams1), 1)
            sent2_linear = torch.cat((sent2_linear, trigrams2), 1)

        return sent1_linear, sent2_linear

class atten(nn.Module):
    '''
        intra sentence attention
    '''

    def __init__(self, hidden_size, dropout):
        super(atten, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.mlp_f = self._mlp_layers(self.hidden_size, self.hidden_size)
        self.mlp_g = self._mlp_layers(2 * self.hidden_size, self.hidden_size)
        self.mlp_h = self._mlp_layers(2 * self.hidden_size, self.hidden_size)
        # self.mlp_h = self._mlp_layers(2 * self.hidden_size + 2 * 2 * self.hidden_size, self.hidden_size) # highway connection

        self.final_linear = nn.Linear(self.hidden_size, 1, bias=True)

        self.sigmoid = nn.Sigmoid()

        '''initialize parameters'''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight)


    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=self.dropout))
        mlp_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())        
        #mlp_layers.append(nn.Dropout(p=self.dropout))
        #mlp_layers.append(nn.Linear(output_dim, output_dim, bias=True))
        #mlp_layers.append(nn.ReLU())        
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1_linear, sent2_linear):
        '''
            sent_linear: batch_size x length x hidden_size
        '''
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)

        
        '''attend'''
        f1 = self.mlp_f(sent1_linear.view(-1, self.hidden_size))
        f2 = self.mlp_f(sent2_linear.view(-1, self.hidden_size))

        f1 = f1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        # batch_size x len1 x len2
        
        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1)).view(-1, len2, len1)
        # batch_size x len2 x len1

        '''compare'''    
        sent1_combine = torch.cat((sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat((sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.hidden_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.hidden_size))
        g1 = g1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        #sent1_h = torch.squeeze(sent1_h, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)
        #sent2_h = torch.squeeze(sent2_h, 1)

        # Highway connection
        # input_combine = torch.cat((sent1_output * sent2_output, torch.abs(sent1_output - sent2_output), sent1_h * sent2_h, torch.abs(sent1_h - sent2_h) ), 1)

        input_combine = torch.cat((sent1_output * sent2_output, torch.abs(sent1_output - sent2_output)), 1)
        
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        h = self.final_linear(h)

        return 5*self.sigmoid(h)[:,0]
