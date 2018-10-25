'''
baseline model:
    standard intra-attention
    share parameters by default
'''

import argparse
import sys
import time
import logging
import h5py
import random
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from models.baseline_ts import encoder
from models.baseline_ts import atten

from models.sts_data import sts_data_STSBenchmark
from models.sts_data import score2labels
from models.sts_data import labels2score
from models.sts_data import pearson

from models.snli_data import w2v
np.seterr(divide='ignore', invalid='ignore')


def train(args):
            
    if args.max_length < 0:
        args.max_length = 9999

    #####################
    # INITIALIZE LOGGER #
    #####################
    logger_name = "log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(args.log_dir + args.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    ###################
    # SET CUDA DEVICE #
    ###################
    torch.cuda.set_device(args.gpu_id)

    ###################
    # SET RANDOM SEED #
    ###################
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #############
    # LOAD DATA #
    #############
    logger.info('Loading train data...')

    train_data = sts_data_STSBenchmark(args.train_file, args.max_length)
    train_batches = train_data.batches
    train_samples = train_data.label.size(0)

    dev_data = sts_data_STSBenchmark(args.dev_file, args.max_length)
    dev_batches = dev_data.batches
    dev_samples = dev_data.label.size(0)

    test_data = sts_data_STSBenchmark(args.test_file, args.max_length)
    test_batches = test_data.batches
    test_samples = test_data.label.size(0)
        
    logger.info('Train size # sent ' + str(train_data.size))
    logger.info('Dev size # sent ' + str(dev_data.size))
    logger.info('Test size # sent ' + str(test_data.size))

    ###################
    # LOAD EMBEDDINGS #
    ###################
    logger.info('Loading input embeddings...')
    word_vecs = w2v(args.w2v_file).word_vecs 

    best_dev = []   # (epoch, dev_acc)

    ###################
    # BUILD THE MODEL #
    ###################
    input_encoder = encoder(word_vecs.size(0), args.embedding_size, args.hidden_size, args.bigrams, args.trigrams)
    input_encoder.embedding.weight.data.copy_(word_vecs)
    input_encoder.embedding.weight.requires_grad = False
    inter_atten = atten(args.hidden_size, args.dropout)

    input_encoder.cuda()
    inter_atten.cuda()

    para1 = filter(lambda p: p.requires_grad, input_encoder.parameters())
    para2 = inter_atten.parameters()

    #############
    # OPTIMIZER #
    #############
    if args.optimizer == 'Adagrad':
        input_optimizer = optim.Adagrad(para1, lr=args.lr, weight_decay=args.weight_decay)
        inter_atten_optimizer = optim.Adagrad(para2, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        input_optimizer = optim.Adadelta(para1, lr=args.lr)
        inter_atten_optimizer = optim.Adadelta(para2, lr=args.lr)
    elif args.optimizer == 'SGD':
        input_optimizer = optim.SGD(para1, lr=args.lr)
        inter_atten_optimizer = optim.SGD(para2, lr=args.lr)
    elif args.optimizer == 'RMSProp':
        input_optimizer = optim.RMSprop(para1, lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0.05, centered=False)
        inter_atten_optimizer = optim.RMSprop(para2, lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0.05, centered=False)
    elif args.optimizer == 'Adam':
        input_optimizer = optim.Adam(para1, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        inter_atten_optimizer = optim.Adam(para2, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        logger.info('No Optimizer.')
        sys.exit()

    #############
    # CRITERION #
    #############
    # criterion = nn.KLDivLoss()
    # criterion = nn.NLLLoss(size_average=True)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    logger.info('Start to train...')
    for k in range(args.epoch):

        train_predictions = torch.FloatTensor(train_samples)
        train_golds = torch.FloatTensor(train_samples)
        
        total = 0.
        loss_data = 0.
        train_sents = 0.

        shuffle(train_batches)
        timer = time.time()
        
        for i in range(len(train_batches)):
            
            train_src_batch, train_tgt_batch, train_lbl_batch = train_batches[i]

            train_src_batch = Variable(train_src_batch.cuda())
            train_tgt_batch = Variable(train_tgt_batch.cuda())
            train_scores_batch = Variable(train_lbl_batch.float().cuda())
                            
            batch_size = train_src_batch.size(0)        

            # zero the gradients
            input_optimizer.zero_grad()
            inter_atten_optimizer.zero_grad()

            ########################
            # INITIALIZE OPTIMIZER #
            ########################
            # if k == 0 and optim == 'Adagrad':
            #     for group in input_optimizer.param_groups:
            #         for p in group['params']:
            #             state = input_optimizer.state[p]
            #             state['sum'] += args.Adagrad_init
            #     for group in inter_atten_optimizer.param_groups:
            #         for p in group['params']:
            #             state = inter_atten_optimizer.state[p]
            #             state['sum'] += args.Adagrad_init

            # forward samples
            train_src_linear, train_tgt_linear = input_encoder(train_src_batch, train_tgt_batch)
            predict = inter_atten(train_src_linear, train_tgt_linear)
            loss = criterion(predict, train_scores_batch)

            # compute gradients
            loss.backward()

            grad_norm = 1e-8
            para_norm = 1e-8

            for param in filter(lambda p: p.requires_grad, input_encoder.parameters()):
                grad_norm += param.grad.data.norm() ** 2
                para_norm += param.data.norm() ** 2
                
            for param in filter(lambda p: p.requires_grad, inter_atten.parameters()):
                grad_norm += param.grad.data.norm() ** 2
                para_norm += param.data.norm() ** 2

            # update params
            input_optimizer.step()
            inter_atten_optimizer.step()

            loss_data += (loss.data[0] * batch_size)  # / train_lbl_batch.data.size()[0])

            for index in range(int(batch_size)):
                train_predictions[int(index + total)] = predict.data[index]
                train_golds[int(index + total)] = train_scores_batch.data[index]

            total += batch_size
            pearsonCor = pearson(train_predictions[:int(total)], train_golds[:int(total)])
            
            
            if (i + 1) % args.display_interval == 0:                
                logger.info('epoch %d, batches %d|%d, train-pearson %.3f, loss %.3f, para-norm %.3f, grad-norm %.3f, time %.2fs, ' %
                            (k, i + 1, len(train_batches), pearsonCor, loss_data / total, para_norm, grad_norm, time.time() - timer))
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.
                
            if i == len(train_batches) - 1:
                # todo print pearson
                logger.info('epoch %d, batches %d|%d, train-pearson %.3f, loss %.3f, para-norm %.3f, grad-norm %.3f, time %.2fs, ' %
                            (k, i + 1, len(train_batches), pearsonCor, loss_data / total, para_norm, grad_norm, time.time() - timer))
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.           

        ############
        # EVALUATE #
        ############
        if (k + 1) % args.dev_interval == 0:
            input_encoder.eval()
            inter_atten.eval()

            dev_predictions = torch.FloatTensor(dev_samples)
            dev_golds = torch.FloatTensor(dev_samples)
        
            total = 0.

            for i in range(len(dev_batches)):
                dev_src_batch, dev_tgt_batch, dev_lbl_batch = dev_batches[i]

                batch_size = dev_src_batch.size(0)        
                
                dev_src_batch = Variable(dev_src_batch.cuda())
                dev_tgt_batch = Variable(dev_tgt_batch.cuda())
                dev_scores_batch = Variable(dev_lbl_batch.cuda())
                                
                dev_src_linear, dev_tgt_linear = input_encoder(dev_src_batch, dev_tgt_batch)
                predict = inter_atten(dev_src_linear, dev_tgt_linear).data

                for index in range(int(batch_size)):
                    dev_predictions[int(index + total)] = predict[index]
                    dev_golds[int(index + total)] = dev_scores_batch.data[index]

                total += batch_size

            dev_acc = pearson(dev_predictions, dev_golds)
            logger.info('dev-pearson %.3f' % (dev_acc))

            
            if (k + 1) / args.dev_interval == 1:
                model_fname = '%s%s_epoch-%d_dev-pearson-%.3f' %(args.model_path, args.log_fname.split('.')[0], k, dev_acc)
                torch.save(input_encoder.state_dict(), model_fname + '_input-encoder.pt')
                torch.save(inter_atten.state_dict(), model_fname + '_inter-atten.pt')
                best_dev.append((k, dev_acc, model_fname))
                logger.info('current best-dev:')
                
                for t in best_dev:
                    logger.info('\t%d %.3f' %(t[0], t[1]))
                logger.info('save model!')
                
            else:
                if dev_acc > best_dev[-1][1]:
                    model_fname = '%s%s_epoch-%d_dev-pearson-%.3f' %(args.model_path, args.log_fname.split('.')[0], k, dev_acc)
                    torch.save(input_encoder.state_dict(), model_fname + '_input-encoder.pt')
                    torch.save(inter_atten.state_dict(), model_fname + '_inter-atten.pt')
                    best_dev.append((k, dev_acc, model_fname))
                    logger.info('current best-dev:')
                    
                    for t in best_dev:
                        logger.info('\t%d %.3f' %(t[0], t[1]))
                        
                    logger.info('save model!') 

            input_encoder.train()
            inter_atten.train()

    logger.info('training end!')
    
    ########
    # TEST #
    ########
    best_model_fname = best_dev[-1][2]
    input_encoder.load_state_dict(torch.load(best_model_fname + '_input-encoder.pt'))
    inter_atten.load_state_dict(torch.load(best_model_fname + '_inter-atten.pt'))

    input_encoder.eval()
    inter_atten.eval()

    test_predictions = torch.FloatTensor(test_samples)
    test_golds = torch.FloatTensor(test_samples)
    
    total = 0.

    results_out = open(model_fname + 'results.txt','w')

    for i in range(len(test_batches)):
        test_src_batch, test_tgt_batch, test_lbl_batch = test_batches[i]

        batch_size = test_src_batch.size(0)        
        
        test_src_batch = Variable(test_src_batch.cuda())
        test_tgt_batch = Variable(test_tgt_batch.cuda())
        test_scores_batch = Variable(test_lbl_batch.cuda())
        
        test_src_linear, test_tgt_linear = input_encoder(test_src_batch, test_tgt_batch)
        predict = inter_atten(test_src_linear, test_tgt_linear).data

        for index in range(int(batch_size)):
            test_predictions[int(index + total)] = predict[index]
            test_golds[int(index + total)] = test_scores_batch.data[index]
            results_out.write("{} {}\n".format(predict[index], test_scores_batch.data[index]))
        
        total += batch_size                
        
    test_acc = pearson(test_predictions, test_golds)
    logger.info('test-pearson %.3f' % (test_acc))

    results_out.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_file', help='training data file (hdf5)', type=str, default='entail-train.hdf5')
    parser.add_argument('--dev_file', help='development data file (hdf5)', type=str, default='entail-val.hdf5')
    parser.add_argument('--test_file', help='test data file (hdf5)', type=str, default='entail-test.hdf5')

    parser.add_argument('--bigrams', help='compute bigrams', default=False, action='store_true')
    parser.add_argument('--trigrams', help='compute trigrams', default=False, action='store_true')

    parser.add_argument('--dropout', help='Dropout value pr', type=float, default=0.0)
    
    parser.add_argument('--w2v_file', help='pretrained word vectors file (hdf5)', type=str, default='glove.hdf5')
    parser.add_argument('--embedding_size', help='word embedding size', type=int, default=300)
    
    parser.add_argument('--log_dir', help='log file directory', type=str, default='.')
    parser.add_argument('--log_fname', help='log file name', type=str, default='log_snli.log')

    parser.add_argument('--gpu_id', help='GPU device id', type=int, default=1)

    parser.add_argument('--epoch', help='training epoch', type=int, default=250)
    parser.add_argument('--dev_interval', help='interval for development', type=int, default=1)
    parser.add_argument('--optimizer', help='optimizer', type=str, default='Adagrad')
    parser.add_argument('--Adagrad_init', help='initial accumulating values for gradients', type=float, default=0.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.05)
    parser.add_argument('--hidden_size', help='hidden layer size', type=int, default=300)
    parser.add_argument('--max_length', help='maximum length of training sentences, -1 means no length limit', type=int, default=-1)
    parser.add_argument('--display_interval', help='interval of display', type=int, default=1000)

    parser.add_argument('--weight_decay', help='l2 regularization', type=float, default=5e-5)
    parser.add_argument('--model_path', help='path of model file (not include the name suffix', type=str, default='.')
    parser.add_argument('--seed', help='random seed', type=int, default=0)

    args=parser.parse_args()
    # args.max_lenght = 10   # args can be set manually like this
    train(args)

else:
    pass
