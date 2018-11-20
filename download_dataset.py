# -*- coding: utf-8 -*-
# Original Code : https://github.com/alrojo/CB513/blob/master/data.py

import os
import numpy as np
import subprocess
from utils import load_gz
from features import prot_to_vector

TRAIN_PATH = '../data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = '../data/cb513+profile_split1.npy.gz'
DATASET_PATH = '../data/cb513.npz'

# convert onehot to string for both AA and SS
# NOTE: the ordering of these comes from the dataset readme
aa_list = ["A", "C", "E", "D", "G", "F", "I", "H", "K", "M", "L", "N", "Q",
           "P", "S", "R", "T", "W", "V", "Y", "X", "NoSeq", "<s>", "</s>"]
ss_list = ["L", "B", "E", "G", "I", "H", "S", "T", "NoSeq", "<s>", "</s>"]


def download_dataset():
    #print('Download CB513 dataset ...')
    os.makedirs('../data/', exist_ok=True)
    #subprocess.call("./download_dataset.sh", shell=True)
    X_train, y_train, seq_len_train = get_train()
    X_test, y_test, seq_len_test = get_test()
    X = np.append(X_train, X_test, axis=0)
    y = np.append(y_train, y_test, axis=0)
    seq_len = np.append(seq_len_train, seq_len_test, axis=0)
    np.savez_compressed(DATASET_PATH, X=X, y=y, seq_len=seq_len)
    print(f'Saved CB513 dataset in {DATASET_PATH}')
    
def add_features(data):
    num_samples = data.shape[0]     
    # NOTE: convert one-hot sequence to string
    seqs_inds = np.argmax(data[:, :, 0:22], axis=2)
    ss_inds = np.argmax(data[:, :, 22:31], axis=2)
    seqs = []
    ss = []
    for i in range(seqs_inds.shape[0]):
        # convert the indices to letters, ignoring noseqs
        seq = "".join([aa_list[seqs_inds[i,j]] for j in range(seqs_inds.shape[1]) if aa_list[seqs_inds[i,j]] != "NoSeq"])
        ss_labels = "".join([ss_list[ss_inds[i,j]] for j in range(ss_inds.shape[1]) if ss_list[ss_inds[i,j]] != "NoSeq"])
        try:
            assert len(seq) == len(ss_labels)
        except:
            print(seq)
            print(ss_labels)
            raise
        seqs.append(seq)
        ss.append(ss_labels) 
        
    new_data = []
    for index in range(num_samples):       
        more_features = prot_to_vector(seqs[index])       
        results = np.zeros(((data[index]).shape[0], more_features.shape[1]), dtype='float32')
        results[:more_features.shape[0],:more_features.shape[1]] = more_features
        total_features = np.concatenate((data[index], results), axis=1)
        new_data.append(total_features)    
    more_features_num = more_features.shape[1]
    new_data = np.array(new_data)
    
    try:
        assert new_data.shape[2] == data.shape[2] + more_features_num
    except:
        print("something wrong during adding features")
        raise  

    return new_data, more_features_num

def update_patch(data):
    num_samples = data.shape[0]     
    # NOTE: convert one-hot sequence to string
    seqs_inds = np.argmax(data[:, :, 0:22], axis=2)
    ss_inds = np.argmax(data[:, :, 22:31], axis=2)

    print(seqs_inds[0])
    print(ss_inds[0])

    seqs = []
    ss = []
    for i in range(seqs_inds.shape[0]):
        # convert the indices to letters, ignoring noseqs
        seq = [aa_list[seqs_inds[i,j]] for j in range(seqs_inds.shape[1]) if aa_list[seqs_inds[i,j]] != "NoSeq"]
        ss_labels = [ss_list[ss_inds[i,j]] for j in range(ss_inds.shape[1]) if ss_list[ss_inds[i,j]] != "NoSeq"]
        seq.insert(0,"<s>")
        seq.append("</s>")
        ss_labels.insert(0,"<s>")
        ss_labels.append("</s>")

        try:
            assert len(seq) == len(ss_labels)
        except:
            print(seq)
            print(ss_labels)
            raise
        seqs.append(seq)
        ss.append(ss_labels)
    print(seqs[0])
    print(ss[0]) 

def get_():
    X_in = load_gz(TRAIN_PATH)
    X = np.reshape(X_in,(5534,700,57))
    del X_in
    X = X[:,:,:]
    labels = X[:,:,22:30]
    mask = X[:,:,30] * -1 + 1

    # b = np.arange(35,56)
    X = X[:,:,35:56]
    return X

def get_train():
    X_in = load_gz(TRAIN_PATH)
    X = np.reshape(X_in,(5534,700,57))
    del X_in
    X = X[:,:,:]

    #X, new_features_num = add_features(X)
    update_patch(X)  
    
    labels = X[:,:,22:30]
    mask = X[:,:,30] * -1 + 1

    a = np.arange(0,21)
    b = np.arange(35,56)
    #f = np.arange(57,57+new_features_num)
    #c = np.hstack((a,b,f))
    c = np.hstack((a,b))
    X = X[:,:,c]

    print("train dataset shape: ", X.shape)
    #print("new features number: ", new_features_num) 

    # getting meta
    num_seqs = np.size(X,0)
    seqlen = np.size(X,1)
    d = np.size(X,2)
    num_classes = 8

    #### REMAKING LABELS ####
    X = X.astype('float32')
    mask = mask.astype('float32')

    print(X[0].shape)
    print(len(mask[0]))
    #print(X[0][314], X[0][315])
    #print(mask[0][314],mask[0][315] )

    exit()


    # Dummy -> concat
    vals = np.arange(0,8)
    labels_new = np.zeros((num_seqs,seqlen))
    for i in range(np.size(labels,axis=0)):
        labels_new[i,:] = np.dot(labels[i,:,:], vals)
    labels_new = labels_new.astype('float32')
    labels = labels_new

    # print("Loading splits ...")
    ##### SPLITS #####
    # getting splits (cannot run before splits are made)
    #split = np.load("data/split.pkl")

    seq_names = np.arange(0,num_seqs)
    #np.random.shuffle(seq_names)

    X_train = X[seq_names[0:5534]]
    X_train = X_train.transpose(0, 2, 1)
    # X_valid = X[seq_names[5278:5534]]
    labels_train = labels[seq_names[0:5534]]
    # labels_valid = labels[seq_names[5278:5534]]
    mask_train = mask[seq_names[0:5534]]
    seq_len_train = mask_train.sum(axis=1)
    # mask_valid = mask[seq_names[5278:5534]]
    num_seq_train = np.size(X_train,0)
    # num_seq_valid = np.size(X_valid,0)
    return X_train, labels_train, seq_len_train


def get_test():
    X_test_in = load_gz(TEST_PATH)
    X_test = np.reshape(X_test_in,(514,700,57))
    del X_test_in
    X_test = X_test[:,:,:].astype('float32')
    
    #X_test, new_features_num = add_features(X_test)
    
    labels_test = X_test[:,:,22:30]
    mask_test = X_test[:,:,30].astype('float32') * -1 + 1

    a = np.arange(0,21)
    b = np.arange(35,56)
    #f = np.arange(57,57+new_features_num)
    #c = np.hstack((a,b,f))
    c = np.hstack((a,b))
    X_test = X_test[:,:,c]
    
    print("test dataset shape: ", X_test.shape)
    #print("new features number: ", new_features_num)     

    # getting meta
    seqlen = np.size(X_test,1)
    d = np.size(X_test,2)
    num_classes = 8
    num_seq_test = np.size(X_test,0)
    X_test = X_test.transpose(0, 2, 1)
    #del a, b, f, c

    ## DUMMY -> CONCAT ##
    vals = np.arange(0,8)
    labels_new = np.zeros((num_seq_test,seqlen))
    for i in range(np.size(labels_test,axis=0)):
        labels_new[i,:] = np.dot(labels_test[i,:,:], vals)
    labels_new = labels_new.astype('float32')
    labels_test = labels_new
    seq_len_test = mask_test.sum(axis=1)

    ### ADDING BATCH PADDING ###

    # X_add = np.zeros((126,seqlen,d))
    # label_add = np.zeros((126,seqlen))
    # mask_add = np.zeros((126,seqlen))
    #
    # X_test = np.concatenate((X_test,X_add), axis=0).astype(theano.config.'float32'X)
    # labels_test = np.concatenate((labels_test, label_add), axis=0).astype('int32')
    # mask_test = np.concatenate((mask_test, mask_add), axis=0).astype(theano.config.'float32'X)
    return X_test, labels_test, seq_len_test
