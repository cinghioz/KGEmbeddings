#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, all_triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.all_triples = all_triples
        # self.triple_set = set(map(tuple, all_triples))
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = count_frequency(triples)
        self.true_head, self.true_tail = get_true_head_and_tail(self.all_triples)
        
    def __len__(self):
        return self.len

    # def precompute_negatives(self, oversample_factor: int = 10):
    #     """
    #     Precompute negative samples for all triples.
    #     Stores them in self.negative_samples as a LongTensor.
    #     Shape: (len(triples), negative_sample_size, 2)
    #     """
    #     n_triples = len(self.triples)
    #     negatives = np.zeros((n_triples, self.negative_sample_size), dtype=np.int32)

    #     # Pre-generate large candidate pools
    #     candidate_e = np.random.randint(
    #         self.nentity,
    #         size=(n_triples, oversample_factor * self.negative_sample_size)
    #     )
    #     # candidate_r = np.random.randint(
    #     #     self.nrelation,
    #     #     size=(n_triples, oversample_factor * self.negative_sample_size)
    #     # )

    #     for idx, (head, relation, tail) in enumerate(self.triples):
    #         neg_pairs = []
    #         check_neg_cands = 0
    #         cand_e = candidate_e[idx]

    #         while check_neg_cands < self.negative_sample_size:
    #             if self.mode == 'head-batch':
    #                 mask = ~np.isin(cand_e, self.true_head[(relation, tail)], assume_unique=True)
    #             elif self.mode == 'tail-batch':
    #                 mask = ~np.isin(cand_e, self.true_tail[(head, relation)], assume_unique=True)
    #             else:
    #                 raise ValueError(f'Training batch mode {self.mode} not supported')

    #             if np.count_nonzero(mask) > 0:
    #                 neg_pairs.extend(cand_e[mask][:self.negative_sample_size - check_neg_cands].tolist())
    #                 check_neg_cands += np.count_nonzero(mask)

    #             cand_e = np.random.randint(self.nentity, size=self.negative_sample_size)

    #         # Trim to required size
    #         negatives[idx] = neg_pairs[:self.negative_sample_size]

    #     self.negative_samples = torch.from_numpy(np.array(negatives))
    
    # def __getitem__(self, idx):
    #     positive_sample = self.triples[idx]
    #     head, relation, tail = positive_sample

    #     subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
    #     subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

    #     # fetch precomputed negatives
    #     if self.negative_samples is None:
    #         raise RuntimeError("Call `precompute_negatives()` before using dataset")

    #     negative_sample = self.negative_samples[idx]

    #     positive_sample = torch.LongTensor(positive_sample)

    #     return positive_sample, negative_sample, subsampling_weight, self.mode

    def __getitem__(self, idx): 
        positive_sample = self.triples[idx] 
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample_e = np.random.randint(self.nentity, size=self.negative_sample_size)
            negative_sample_r = np.random.randint(self.nrelation, size=self.negative_sample_size)
            # negative_sample = np.stack((negative_sample_e, negative_sample_r), axis=1)
            negative_sample = negative_sample_e

            if self.mode == 'head-batch':
                mask = ~np.isin(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True
                )
            elif self.mode == 'tail-batch':
                mask = ~np.isin(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(map(tuple, all_true_triples))
        self.triples = triples
        self.all_true_triples = all_true_triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        # self.true_head, self.true_tail = get_true_head_and_tail(self.all_true_triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


@staticmethod
def count_frequency(triples, start=4):
    '''
    Get frequency of a partial triple like (head, relation) or (relation, tail)
    The frequency will be used for subsampling like word2vec
    '''
    count = {}
    for head, relation, tail in triples:
        if (head, relation) not in count:
            count[(head, relation)] = start
        else:
            count[(head, relation)] += 1

        if (tail, -relation-1) not in count:
            count[(tail, -relation-1)] = start
        else:
            count[(tail, -relation-1)] += 1
    return count

@staticmethod
def get_true_head_and_tail(triples):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''
    
    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

    return true_head, true_tail