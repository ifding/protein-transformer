''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants

import os
import numpy as np
from utils import load_gz

TRAIN_PATH = '../data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = '../data/cb513+profile_split1.npy.gz'


# convert onehot to string for both AA and SS
# NOTE: the ordering of these comes from the dataset readme
aa_list = ["A", "C", "E", "D", "G", "F", "I", "H", "K", "M", "L", "N", "Q",
           "P", "S", "R", "T", "W", "V", "Y", "X", "NoSeq"]
ss_list = ["L", "B", "E", "G", "I", "H", "S", "T", "NoSeq"]

src_list = ["<blank>", "<unk>", "<s>", "</s>", "A", "C", "E", "D", "G", "F", "I", "H", "K", "M", "L", "N", "Q", "P", "S", "R", "T", "W", "V", "Y", "X"]
tgt_list = ["<blank>", "<unk>", "<s>", "</s>", "L", "B", "E", "G", "I", "H", "S", "T"]

src_word2idx = {k: v for v, k in enumerate(src_list)}
tgt_word2idx = {k: v for v, k in enumerate(tgt_list)}


# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def chunks(src, tgt, n):
    """ Yield n successive chunks from l.
    """
    newn = int(1.0 * len(src) / n + 0.5)
    for i in range(0, n-1):
        yield (src[i*newn:i*newn+newn], tgt[i*newn:i*newn+newn])
    yield (src[n*newn-newn:], tgt[n*newn-newn:])
    

def get_instances_from_file(data):
    num_samples = data.shape[0]     
    # NOTE: convert one-hot sequence to string
    seqs_inds = np.argmax(data[:, :, 0:22], axis=2)
    ss_inds = np.argmax(data[:, :, 22:31], axis=2)
    mask = data[:,:,30] * -1 + 1
    
    b = np.arange(35,56)
    pssm = data[:,:,b].tolist()

    seqs = []
    ss = []
    for i in range(seqs_inds.shape[0]):
        # convert the indices to letters, ignoring noseqs
        seq = [aa_list[seqs_inds[i,j]] for j in range(seqs_inds.shape[1]) if aa_list[seqs_inds[i,j]] != "NoSeq"]
        ss_labels = [ss_list[ss_inds[i,j]] for j in range(ss_inds.shape[1]) if ss_list[ss_inds[i,j]] != "NoSeq"]
        
        # convert letters to the indices
        seq = [src_word2idx[i] for i in seq]
        ss_labels = [tgt_word2idx[i] for i in ss_labels]
                
        real_len = len(seq)
        feats = pssm[i][0:real_len]
        
        # concatenate aa type and pssm together
        seq = [[x]+y for x,y in zip(seq, feats)]
        
        dim = np.size(seq, 1)

        try:
            assert len(seq) == len(ss_labels) and len(ss_labels) == sum(mask[i])
        except:
            print(seq)
            print(ss_labels)
            raise              
        
        if len(seq) <= 100:          
            # add bos and eos for aa seq and ss labels
            seq.insert(0,[src_list.index("<s>")]*dim)
            seq.append([src_list.index("</s>")]*dim)
            ss_labels.insert(0,tgt_list.index("<s>"))
            ss_labels.append(tgt_list.index("</s>"))

            seqs.append(seq)
            ss.append(ss_labels)
        else:
            chunks_num = len(seq)//100 + 1
            src_tgt_chunks = chunks(seq, ss_labels, chunks_num)
            for (src, tgt) in src_tgt_chunks:                
                # add bos and eos for aa seq and ss labels
                src.insert(0,[src_list.index("<s>")]*dim)
                src.append([src_list.index("</s>")]*dim)
                tgt.insert(0,tgt_list.index("<s>"))
                tgt.append(tgt_list.index("</s>"))

                seqs.append(src)
                ss.append(tgt)

    return np.array(seqs), np.array(ss)


def get_train():
    print('[Info] Build training and validation dataset ...')
    X_in = load_gz(TRAIN_PATH)
    X = np.reshape(X_in,(5534,700,57))
    del X_in
    X = X[:,:,:]
    
    X_total, labels_total = get_instances_from_file(X)
    
    # getting meta
    num_seqs = np.size(X_total,0)
    seq_names = np.arange(0,num_seqs)
    
    print('[Info] Num of training and validation dataset: ', num_seqs)
    
    #boundary = 5278
    boundary = int(0.9*num_seqs)

    X_train = X_total[seq_names[0:boundary]]
    labels_train = labels_total[seq_names[0:boundary]]
    
    X_valid = X_total[seq_names[boundary:num_seqs]]
    labels_valid = labels_total[seq_names[boundary:num_seqs]]

    return X_train.tolist(),labels_train.tolist(),X_valid.tolist(),labels_valid.tolist()

def get_test():
    print('[Info] Build testing dataset ...')
    X_test_in = load_gz(TEST_PATH)
    X_test = np.reshape(X_test_in,(514,700,57))
    del X_test_in
    X_test = X_test[:,:,:]
    
    X_test, labels_test = get_instances_from_file(X_test)
    
    print('[Info] Num of test dataset: ', np.size(X_test,0))

    return X_test.tolist(),labels_test.tolist()

def main():    
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-save_data', type=str, default='../data/profile.6133.filtered.pt')
    parser.add_argument('-max_len', '--max_seq_len', type=int, default=100)

    opt = parser.parse_args()
    #opt.max_token_seq_len = opt.max_seq_len + 2 # include the <s> and </s> 
    opt.max_token_seq_len = opt.max_seq_len + 4 # include '<blank>', '<unk>', '<s>' and '</s>'   
           
    os.makedirs('../data/', exist_ok=True)
    train_src_insts, train_tgt_insts, valid_src_insts, valid_tgt_insts = get_train()


    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}
    
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')
    


    
    
    
    
    
    
    
    
    
def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]
    
    # word_insts:   2d list
    # [['<s>', '5', 'athletes', 'wearing', 'white', 'tops', 'and', 'white', 'bottoms', 
    # 'are', 'doing', 'a', 'routine', 'for', 'an', 'audience', '.', '</s>'], 
    # ['<s>', 'construction', 'workers', 'standing', 'outside', 'looking', 'at', 
    # 'wires', 'above', 'them', '.', '</s>'] ... ]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}
    
    word_count = {w: 0 for w in full_vocab}
    
    for sent in word_insts:
        for word in sent:
            word_count[word] += 1      
            
    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    
    # word2idx: dict
    # {'<s>': 2, '</s>': 3, '<blank>': 0, '<unk>': 1, 'pfeife': 4, 'nachdem': 5 ... }
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main_():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)
    
    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)
    
    print(valid_src_insts)
    
    print(valid_tgt_insts)
    
    exit()

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
