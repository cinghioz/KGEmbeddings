#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import torch
import os
from collections import defaultdict

class TripletsEngine:
    def __init__(self, path, from_splits=False, feat_path=None):
        self.path = path
        self.from_splits = from_splits
        self.feat_path = feat_path
        self.triplets = np.array([], dtype=object)
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.number_of_entities = 0
        self.number_of_relations = 0
        if from_splits:
            self._load_triplets_from_splits(
            os.path.join(path, "train.txt"),
            os.path.join(path, "valid.txt"),
            os.path.join(path, "test.txt")
        )
        elif os.path.isdir(self.path) and os.path.isfile(self.path+'/entities.txt') and os.path.isfile(self.path+'/relations.txt'):
                self._load_triplets_proc()
        else:
                self._load_triplets()
        if feat_path:
            self._load_features()
        self._generate_mappings()
    
    def _load_triplets_proc(self):
        df = pd.read_csv(self.path+'/triplets.txt', sep='\t', header=None)
        self.triplets = df.values
        self.number_of_relations = len(self.triplets)

        ents = pd.read_csv(self.path+'/entities.txt', sep='\t', header=None, names=['id', 'name'])
        self.entity_to_id = ents.set_index('name')['id'].to_dict()
        self.number_of_entities = len(self.entity_to_id)

        rels = pd.read_csv(self.path+'/relations.txt', sep='\t', header=None, names=['id', 'name'])
        self.relation_to_id = rels.set_index('name')['id'].to_dict()

    def _load_triplets_from_splits(self, train_file, valid_file, test_file):
        # Load files
        train_df = pd.read_csv(train_file, sep='\t', header=None)
        valid_df = pd.read_csv(valid_file, sep='\t', header=None)
        test_df  = pd.read_csv(test_file, sep='\t', header=None)

        all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        self.triplets = all_df.values

        self._index_triplets()

        train_size = len(train_df)
        valid_size = len(valid_df)

        self.train_set = np.arange(0, train_size)
        self.valid_set = np.arange(train_size, train_size + valid_size)
        self.test_set  = np.arange(train_size + valid_size, len(self.triplets))

    def _load_triplets(self):
        if self.path.endswith('.csv'):
            self._load_triplets_csv()
        elif self.path.endswith('.txt'):
            self._load_triplets_txt()
        else:
            raise ValueError("Unsupported file format. Use .csv or .txt")
        
        self._index_triplets()
        
    def _load_triplets_csv(self):
        df = pd.read_csv(self.path)
        self.triplets = df.values

    def _load_triplets_txt(self):
        df = pd.read_csv(self.path, sep='\t')
        self.triplets = df.values

    def _index_triplets(self):
        unique_entities = np.unique(np.concatenate((self.triplets[:, 0], self.triplets[:, 2])))
        unique_relations = np.unique(self.triplets[:, 1])

        self.number_of_entities = len(unique_entities)
        self.number_of_relations = len(unique_relations)

        self.entity_to_id = {entity: i for i, entity in enumerate(unique_entities)}
        self.relation_to_id = {relation: i for i, relation in enumerate(unique_relations)}

        # Look up the indices for heads, relations, and tails in their respective sorted arrays
        head_ids = np.searchsorted(unique_entities, self.triplets[:, 0])
        relation_ids = np.searchsorted(unique_relations, self.triplets[:, 1])
        tail_ids = np.searchsorted(unique_entities, self.triplets[:, 2])

        self.triplets = np.column_stack((head_ids, relation_ids, tail_ids))

    def _load_features(self):
        self.node_features = torch.load(self.feat_path)

    def _generate_mappings(self):
        # (head, relation) -> list of tails
        self.h2t = defaultdict(list)

        # (relation, tail) -> list of heads
        self.r2h = defaultdict(list)

        self.indexing_dict = defaultdict(dict)

        for triple in self.triplets:
            head, relation, tail = tuple(triple)
            self.h2t[(head, relation)].append(tail)
            self.r2h[(tail, relation)].append(head)

            if head not in self.indexing_dict:
                self.indexing_dict[head] = {'in': np.empty((0, 2), dtype=np.int64),
                                    'out': np.empty((0, 2), dtype=np.int64),
                                    'count': 0}
            if tail not in self.indexing_dict:
                self.indexing_dict[tail] = {'in': np.empty((0, 2), dtype=np.int64),
                                    'out': np.empty((0, 2), dtype=np.int64),
                                    'count': 0}

            self.indexing_dict[head]['out'] = np.vstack([self.indexing_dict[head]['out'], [tail, relation]])
            self.indexing_dict[head]['count'] += 1

            self.indexing_dict[tail]['in']  = np.vstack([self.indexing_dict[tail]['in'], [head, relation]])
            self.indexing_dict[tail]['count'] += 1

    def split_triplets(self, train_ratio=0.7, valid_ratio=0.10, inductive=False, seed=42):
        np.random.seed(seed)
        triplets_to_split = self.triplets
        self.inductive_set = np.array([]) 

        if inductive:
            unique_entities = np.array(list(self.entity_to_id.values()))
            entity_count = len(unique_entities)
            inductive_entity_count = max(1, int(entity_count * 0.02)) # Ensure at least 1, take 2% of entities

            inductive_entities = np.random.choice(unique_entities, size=inductive_entity_count, replace=False)

            head_is_inductive = np.isin(self.triplets[:, 0], inductive_entities)
            tail_is_inductive = np.isin(self.triplets[:, 2], inductive_entities)
            
            # Use logical OR to get a mask for any triplet containing an inductive entity.
            inductive_mask = np.logical_or(head_is_inductive, tail_is_inductive)

            self.inductive_set = self.triplets[inductive_mask]
            triplets_to_split = self.triplets[~inductive_mask]

            print(f"Held out {len(inductive_entities)} entities.")
            print(f"Moved {len(self.inductive_set)} triplets to the inductive set.")
            print(f"{len(triplets_to_split)} triplets remain for train/valid/test.")
            print("-" * 20)

        total_triplets = len(triplets_to_split)
        indices = np.random.permutation(total_triplets)

        train_size = int(total_triplets * train_ratio)
        valid_size = int(total_triplets * valid_ratio)

        self.train_set = indices[:train_size]
        self.valid_set = indices[train_size:train_size + valid_size]
        self.test_set = indices[train_size + valid_size:]

        del triplets_to_split

        return len(self.train_set), len(self.valid_set), len(self.test_set)
    
    def generate_negative_samples(self, n, positive_batch = np.array([]), part_to_corrupt='head'):
        existing_set = {tuple(triplet) for triplet in self.triplets}
        
        if positive_batch.size > 0:
            base_triplets = np.array(positive_batch)
            n = len(base_triplets)
        else:
            source_indices = np.random.randint(0, len(self.triplets), size=n)
            base_triplets = self.triplets[source_indices]

        random_replacements = np.random.randint(0, self.number_of_entities, size=n)
        corrupted_triplets = base_triplets.copy()
        
        if part_to_corrupt == 'head':
            col_idx = 0
        elif part_to_corrupt == 'tail':
            col_idx = 2
        else:
            raise ValueError("part_to_corrupt must be 'head' or 'tail'.")

        for i in range(n):
            # Initial corruption with the pre-generated random replacement.
            corrupted_triplets[i, col_idx] = random_replacements[i]
            
            # This inner loop handles collisions: it re-samples only if the corruption is invalid.
            # This happens rarely, so it doesn't slow down the process significantly.
            while (tuple(corrupted_triplets[i]) in existing_set or 
                corrupted_triplets[i, col_idx] == base_triplets[i, col_idx]):
                
                # Generate a new random replacement and try again.
                corrupted_triplets[i, col_idx] = np.random.randint(0, self.number_of_entities)

        return corrupted_triplets