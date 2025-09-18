import sys
sys.path.insert(0, '/home/cc/phd/KGEmbeddings/codes')

import numpy as np
import torch
import os
from collections import defaultdict
import random
from tqdm import tqdm
import pickle 
from codes.triplets import TripletsEngine

DICTS_DIR = '/home/cc/phd/KGEmbeddings/data/primekg/'

random.seed(42)

# TODO: Ci sono delle query che non hanno struttura corretta, non so perchè. Controlla la f
def find_queries(h2t, indexing_dict, n_pairs=3, n_queries=10000):
    queries = []
    results = []
    cnt = 0
    for node in indexing_dict.keys():
        if indexing_dict[node]['count'] < 4 or indexing_dict[node]['count'] > 500:
            continue

        if len(queries) >= n_queries:
            break

        elements = indexing_dict[node]['in']
        relations = np.unique(elements[:, 1])
        np.random.shuffle(relations)

        query = []
        result = []
        
        pairs = np.array(np.meshgrid(relations, relations)).T.reshape(-1, 2)

        # Remove same-element pairs if you only want different values
        pairs = pairs[pairs[:,0] != pairs[:,1]]
        np.random.shuffle(pairs)

        for pair in pairs[:min(n_pairs, len(pairs))]:
            try:
                # r1, r2 = relations[:2]
                r1, r2 = pair
                h1 = elements[elements[:, 1] ==  r1].squeeze()[0]
                h2 = elements[elements[:, 1] ==  r2].squeeze()[0]

                t1 = h2t[(h1, r1)]
                t2 = h2t[(h2, r2)]

                target_tails = set(t1 + t2 + [node])
                target_tails.discard(node)

                query.append([(h1, r1), (h2, r2)])
                result.append(target_tails)

                acc = np.empty((0, 2), dtype=np.int64)
                for tt in target_tails:
                    acc = np.vstack([acc, indexing_dict[tt]['out']])

                h = acc[:, 0]
                r = acc[:, 1]

                # Condition 1: count rows per relation ---
                unique_r, r_counts = np.unique(r, return_counts=True)
                mask1 = r_counts >= 2   # at least 2 edges

                # Condition 2: count distinct h per relation ---
                # drop duplicates by (r,h)
                unique_rh = np.unique(acc, axis=0)
                _, rh_counts = np.unique(unique_rh[:, 1], return_counts=True)
                mask2 = rh_counts >= 2  # at least 2 different h

                # Align arrays
                valid_r = np.intersect1d(unique_r[mask1], np.unique(unique_rh[:, 1])[mask2])

                if len(valid_r) > 0:
                    chosen_r = valid_r[0]   
                    # or np.random.choice(valid_r)
                    filtered_targets = np.unique(acc[r == chosen_r][:, 0])
                else:
                    continue

                filtered_targets = set(filtered_targets)
                filtered_targets.discard(node)
                
                query.append(chosen_r)
                result.append(set(filtered_targets))

                queries.append(query)
                results.append(result)
            except:
                continue

        if cnt % 5 == 0:
            print(f"Queries generated: {len(queries)} -> remaining nodes: {len(indexing_dict.keys()) - cnt}")
        cnt += 1

    return queries, results

if __name__ == '__main__':
    kg = TripletsEngine(os.path.join(DICTS_DIR), ext="csv", from_splits=True)

    # (head, relation) -> list of tails
    h2t = defaultdict(list)

    # (relation, tail) -> list of heads
    r2h = defaultdict(list)

    indexing_dict = defaultdict(dict)

    print("Indexing Start.")

    for triple in kg.triplets:
        head, relation, tail = tuple(triple)
        h2t[(head, relation)].append(tail)
        r2h[(tail, relation)].append(head)

        if head not in indexing_dict:
            indexing_dict[head] = {'in': np.empty((0, 2), dtype=np.int64),
                                'out': np.empty((0, 2), dtype=np.int64),
                                'count': 0}
        if tail not in indexing_dict:
            indexing_dict[tail] = {'in': np.empty((0, 2), dtype=np.int64),
                                'out': np.empty((0, 2), dtype=np.int64),
                                'count': 0}

        indexing_dict[head]['out'] = np.vstack([indexing_dict[head]['out'], [tail, relation]])
        indexing_dict[head]['count'] += 1

        indexing_dict[tail]['in']  = np.vstack([indexing_dict[tail]['in'], [head, relation]])
        indexing_dict[tail]['count'] += 1

    print("Indexing done.")

    queries, results = find_queries(h2t, indexing_dict, n_pairs=5, n_queries=10000)

    save_dict = {
        'queries': queries,
        'results': results
    }

    with open('/home/cc/phd/KGEmbeddings/queries/primekg/queries.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    print("Queries generated and saved correctly!")