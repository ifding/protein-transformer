# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.pardir)
import numpy as np
import argparse
#from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim

from data import LoadDataset
from hyperparams import Hyperparams as hp
from transformer import *
from utils import *



from torchtext import data, datasets

# params
# ----------
parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-e', '--epochs', type=int, default=50,
                    help='The number of epochs to run (default: 50)')
parser.add_argument('-b', '--batch_size_train', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
                    help='input batch size for testing (default: 1024)')
parser.add_argument('-k', '--k_fold', type=int, default=10,
                    help='K-Folds cross-validator (default: 10)')
parser.add_argument('--save_dir', type=str, default='../data/result',
                    help='Result path (default: ../data/result)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens



global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def data_gen(V_in, V_out, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        src_data = torch.from_numpy(np.random.randint(1, V_in, size=(batch, 10)))
        tgt_data = torch.from_numpy(np.random.randint(1, V_out, size=(batch, 10)))
        src_data[:, 0] = 1
        tgt_data[:, 0] = 1
        src = Variable(src_data, requires_grad=False)
        tgt = Variable(tgt_data, requires_grad=False)
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)



def main():

    # laod dataset and set k-fold cross validation
    D = LoadDataset(args.batch_size_train, args.batch_size_test)
    idxs = np.arange(D.__len__())
    
    train_idx = np.array(idxs[0:5534])
    #valid_idx = np.array(idxs[5278:5534])
    test_idx = np.array(idxs[5534:6048])
    #test_idx = np.array(idxs[5278:5534])
    
    idx = (train_idx, test_idx)
    train_loader, test_loader = D(idx)


    # Train the simple copy task.
    V_in = 21
    V_out = 8
    criterion = LabelSmoothing(size=V_out, padding_idx=0, smoothing=0.0)
    model = make_model(V_in, V_out, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    for epoch in range(1):
        model.train()
        run_epoch(data_gen(V_in, V_out, 32, 20), model, 
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V_in, V_out, 32, 5), model, 
                        SimpleLossCompute(model.generator, criterion, None)))
    
    model.eval()
    src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
    src_mask = Variable(torch.ones(1, 1, 10) )
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
    



















def mains_():

    # Train the simple copy task.
    V_in = 11
    V_out = 8
    criterion = LabelSmoothing(size=V_in, padding_idx=0, smoothing=0.0)
    model = make_model(V_in, V_in, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V_in, 30, 20), model, 
                SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V_in, 30, 5), model, 
                        SimpleLossCompute(model.generator, criterion, None)))






def train(model, device, train_loader, optimizer, loss_function):
    model.train()
    train_loss = 0
    len_ = len(train_loader)
    for batch_idx, (data, target, seq_len) in enumerate(train_loader):
        data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
        optimizer.zero_grad()
        out = model(data, seq_len)
        loss = loss_function(out, target, seq_len)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(f'\rtrain loss={loss.item()/len(data):.5f}', end=', ')

    train_loss /= len_
    print(f'train loss={train_loss:.5f}', end=', ')

    return train_loss


def evaluate(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    acc = 0
    len_ = len(test_loader)
    with torch.no_grad():
        for i, (data, target, seq_len) in enumerate(test_loader):
            data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
            out = model(data, seq_len)
            test_loss += loss_function(out, target, seq_len).cpu().data.numpy() # sum up batch loss
            acc += accuracy(out, target, seq_len)
            
            ss_outs = ss_output(out, target, seq_len)
            #for i in range(0, len(ss_outs)//2, 2):
            #    print(ss_outs[i])   #target
            #    print(ss_outs[i+1]) #output
            #    print()
            #exit()

    test_loss /= len_
    acc /= len_
    print(f'eval acc={acc:.4f}', end=', ')

    return test_loss, acc, ss_outs


def main_():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # make directory to save train history and model
    #save_dir = f'{args.save_dir}_{timestamp()}'
    #os.makedirs(save_dir, exist_ok=True)

    # laod dataset and set k-fold cross validation
    D = LoadDataset(args.batch_size_train, args.batch_size_test)
    idxs = np.arange(D.__len__())
    
    train_idx = np.array(idxs[0:5534])
    #valid_idx = np.array(idxs[5278:5534])
    test_idx = np.array(idxs[5534:6048])
    #test_idx = np.array(idxs[5278:5534])
    
    idx = (train_idx, test_idx)
    train_loader, test_loader = D(idx)

    # model, loss_function, optimizer
    #model = Net().to(device)
    model = AttentionModel(device).to(device)
    #model = AttModel(hp).to(device)
    loss_function = loss_func
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, betas=[0.9, 0.98], eps=1e-8)

    # train and test
    history = []
    best_eval_acc = None
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, loss_function)
        eval_loss, eval_acc, eval_outs = evaluate(model, device, test_loader, loss_function)
        history.append([train_loss, eval_loss, eval_acc])
        show_progress(epoch, args.epochs)
        
        if not best_eval_acc or eval_acc > best_eval_acc:
            save_model(model, save_dir)
            best_eval_acc = eval_acc
            save_outs(eval_outs, save_dir) # 0: target, 1: output ...

    # save train history and model
    save_history(history, save_dir)


if __name__ == '__main__':
    main()
