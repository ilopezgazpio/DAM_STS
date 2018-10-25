import torch
import numpy as np
import h5py
import math

def score2labels(lbl_batch, lbl_size):
    '''
    input: batch_size * 1 (cpu)
    output: batch_size * label_size (cpu)
    pr distribution over classes from [0..label_size] as in Tai et al 2015
    '''

    batches = lbl_batch.size(0)
    output = torch.FloatTensor(batches, lbl_size).zero_()

    for index, value in enumerate(lbl_batch):
        sim = value
        ceil = int(math.ceil(sim))
        floor = int(math.floor(sim))

        if ceil == floor:
            output[index][floor] = 1
        else:
            output[index][floor] = ceil - sim
            output[index][ceil] = sim - floor

    return output

    
def labels2score(lbl_batch):
    '''
    input: batch_size x label_size
    output: batch_size x 1
    pr distribution over classes from [0..label_size] as in Tai et al 2015
    '''
    lbl_size = lbl_batch.size(1)
    elems = lbl_batch.size(0)
    aux = torch.arange(0, lbl_size, 1).view(1, lbl_size)
    return torch.mm(lbl_batch.data.cpu().exp(), torch.transpose(aux, 0, 1)).view(elems)

    
def pearson(tensor1, tensor2):
    '''
    input:
    tensor1: batch_size * 1
    tensor2: batch_size * 1
    '''

    tensor1 = tensor1.squeeze().numpy()
    tensor2 = tensor2.squeeze().numpy()

    pearson = np.corrcoef(tensor1.T, tensor2.T)[0,1]

    if 0 <= pearson <= 1:
        return pearson
    else:
        # batches of size 1 make corrcoef return np.nan
        return 0


    
class sts_data_SICK(object):
    '''
        class to handle training data
    '''

    def __init__(self, fname, max_length):

        if max_length < 0:
            max_length = 9999

        f = h5py.File(fname, 'r')
        
        self.source = torch.from_numpy(np.array(f['source'])) - 1
        self.target = torch.from_numpy(np.array(f['target'])) - 1
        self.label = torch.from_numpy(np.array(f['label'])) - 1
        self.label_size = torch.from_numpy(np.array(f['label_size']))
        # max target length each batch
        self.source_l = torch.from_numpy(np.array(f['source_l']))
        self.target_l = torch.from_numpy(np.array(f['target_l']))

        # idx in torch style; indicate the start index of each batch (starting  # with 1)
        self.batch_idx = torch.from_numpy(np.array(f['batch_idx'])) - 1
        self.batch_l = torch.from_numpy(np.array(f['batch_l']))

        self.batches = []   # batches
        self.length = self.batch_l.size(0)  # number of batches
        self.size = 0   # number of sentences

        for i in range(self.length):
            if self.source_l[i] <= max_length and self.target_l[i] <= max_length:
              batch = (self.source[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.source_l[i]],
                       self.target[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.target_l[i]],
                       self.label[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]],
                       )
              self.batches.append(batch)
              self.size += self.batch_l[i]

class sts_data_STSBenchmark(object):
    '''
        class to handle training data
    '''

    def __init__(self, fname, max_length):

        if max_length < 0:
            max_length = 9999

        f = h5py.File(fname, 'r')
        
        self.source = torch.from_numpy(np.array(f['source'])) - 1
        self.target = torch.from_numpy(np.array(f['target'])) - 1
        self.label = torch.from_numpy(np.array(f['label']))
        self.label_size = torch.from_numpy(np.array(f['label_size']))
        # max target length each batch
        self.source_l = torch.from_numpy(np.array(f['source_l']))
        self.target_l = torch.from_numpy(np.array(f['target_l']))

        # idx in torch style; indicate the start index of each batch (starting  # with 1)
        self.batch_idx = torch.from_numpy(np.array(f['batch_idx'])) - 1
        self.batch_l = torch.from_numpy(np.array(f['batch_l']))
        self.batch_l[-1] +=1

        self.batches = []   # batches
        self.length = self.batch_l.size(0)  # number of batches
        self.size = 0   # number of sentences

        for i in range(self.length):
            if self.source_l[i] <= max_length and self.target_l[i] <= max_length:
              batch = (self.source[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.source_l[i]],
                       self.target[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.target_l[i]],
                       self.label[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]],
                       )
              self.batches.append(batch)
              self.size += self.batch_l[i]

