#!/usr/bin/env python3
#
# from: https://github.com/cyclone923/SATNET/blob/master/exps/n_queens.py
#

import argparse

import os
import shutil
import csv
import sys

import numpy as np
import numpy.random as npr
#import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import satnet

torch.set_printoptions(linewidth=sys.maxsize)
class NQueensSolver(nn.Module):
    def __init__(self, boardSz, aux, m):
        super(NQueensSolver, self).__init__()
        n = boardSz**2
        self.sat = satnet.SATNet(n, m, aux, max_iter=100, eps=1e-6)

    def forward(self, y_in, mask):
        out = self.sat(y_in, mask)
        return out

class CSVLogger(object):
    def __init__(self, fname):
        self.f = open(fname, 'w')
        self.logger = csv.writer(self.f)

    def log(self, fields):
        self.logger.writerow(fields)
        self.f.flush()

def print_header(msg):
    print('===>', msg)

def find_unperm(perm):
    unperm = torch.zeros_like(perm)
    for i in range(perm.size(0)):
        unperm[perm[i]] = i
    return unperm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='nqueens')
    parser.add_argument('--nQueens', type=int, default=12)
    parser.add_argument('--batchSz', type=int, default=10650 // 2)
    parser.add_argument('--testBatchSz', type=int, default=3550 // 2)
    parser.add_argument('--aux', type=int, default=144)
    parser.add_argument('--m', type=int, default=256)
    parser.add_argument('--nEpoch', type=int, default=200000)
    parser.add_argument('--testPct', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--save', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    # For debugging: fix the random seed
    npr.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda: 
        print('Using', torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()

    save = 'nqueens{}-aux{}-m{}-lr{}-bsz{}'.format(args.nQueens, args.aux, args.m, args.lr, args.batchSz)
    if args.save:
        save = '{}-{}'.format(args.save, save)
    save = os.path.join('logs', save)
    # if os.path.isdir(save):
    #     shutil.rmtree(save)
    os.makedirs(save, exist_ok=True)

    #setproctitle.setproctitle('sudoku.{}'.format(save))

    print_header('Loading data')

    with open(os.path.join(os.path.join(args.data_dir, str(args.nQueens)), 'features.pt'), 'rb') as f:
        X_in = torch.load(f)
    with open(os.path.join(os.path.join(args.data_dir, str(args.nQueens)), 'features.pt'), 'rb') as f:
        Y_in = torch.load(f)
    with open(os.path.join(os.path.join(args.data_dir, str(args.nQueens)), 'is_input.pt'), 'rb') as f:
        is_input = torch.load(f)
    print(is_input[0])
    exit(0)
    # print(Y_in[0])
    # print(is_input[0])
    # exit(0)

    print_header('Forming inputs')
    N = X_in.size(0)
    print(N)
    perm = torch.randperm(N)
    X_in, Y_in, is_input = torch.flatten(X_in[perm], start_dim=1), torch.flatten(Y_in[perm], start_dim=1), torch.flatten(is_input[perm], start_dim=1)


    nTrain = int(N*(1.-args.testPct))
    nTest = N-nTrain
    assert(nTrain % args.batchSz == 0)
    assert(nTest % args.testBatchSz == 0)

    if args.cuda:
        X_in, is_input, Y_in = X_in.cuda(), is_input.cuda(), Y_in.cuda()

    print(X_in.size())
    print(Y_in.size())

    train_set = TensorDataset(X_in[:nTrain], is_input[:nTrain], Y_in[:nTrain])
    test_set =  TensorDataset(X_in[nTrain:], is_input[nTrain:], Y_in[nTrain:])

    model = NQueensSolver(args.nQueens, args.aux, args.m)

    if args.cuda:
        model = model.cuda()


    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.model:
        model.load_state_dict(torch.load(args.model))

    train_logger = CSVLogger(os.path.join(save, 'train.csv'))
    test_logger = CSVLogger(os.path.join(save, 'test.csv'))
    fields = ['epoch', 'loss', 'err']
    train_logger.log(fields)
    test_logger.log(fields)

    # test(0, model, optimizer, test_logger, test_set, args.testBatchSz)
    for epoch in range(1, args.nEpoch+1):
        print(f"Eproch{epoch}")
        train(epoch, model, optimizer, train_logger, train_set, args.batchSz)
        test(epoch, model, optimizer, test_logger, train_set, args.testBatchSz)
        test(epoch, model, optimizer, test_logger, test_set, args.testBatchSz)
        # torch.save(model.state_dict(), os.path.join(save, 'it'+str(epoch)+'.pth'))

def run(epoch, model, optimizer, logger, dataset, batchSz, to_train=False):
    loss_final, err_final = 0, 0
    loader = DataLoader(dataset, batch_size=batchSz, shuffle=True)
    # tloader = tqdm(enumerate(loader), total=len(loader))

    for i,(data,is_input,label) in enumerate(loader):
        if to_train:
            optimizer.zero_grad()
        preds = model(data.contiguous(), is_input.contiguous())
        loss = nn.functional.binary_cross_entropy(preds, label)

        if to_train:
            loss.backward()
            optimizer.step()

        err = computeErr(preds.data, label.data)
        # tloader.set_description('Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err))
        loss_final += loss.item()
        err_final += err

    loss_final, err_final = loss_final/len(loader), err_final/len(loader)
    logger.log((epoch, loss_final, err_final))

    if not to_train:
        print('TESTING ON{}SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(len(dataset), loss_final, err_final))


def train(epoch, model, optimizer, logger, dataset, batchSz):
    run(epoch, model, optimizer, logger, dataset, batchSz, to_train=True)

@torch.no_grad()
def test(epoch, model, optimizer, logger, dataset, batchSz):
    run(epoch, model, optimizer, logger, dataset, batchSz, to_train=False)

@torch.no_grad()
def computeErr(pred_flat, label):
    pred_label = torch.round(pred_flat)
    correct = [ 0 if i.equal(j) else 1 for i,j in zip(pred_label, label)]
    return sum(correct) / len(correct)

if __name__=='__main__':
    main()
