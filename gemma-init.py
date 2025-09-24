import pickle
import json
import os
import torch
import random
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 512
DATA = "umls"

DICTS_DIR = f"/home/cc/phd/KGEmbeddings/data/{DATA}"

# kg = TripletsEngine(DICTS_DIR, from_splits=True, ext='csv')

with open(DICTS_DIR + '/entity_mapping.dict', 'rb') as f:
    umls_map = json.loads(f.read())

with open(DICTS_DIR + '/concept_mapping.json') as f:
    umls_ents = json.loads(f.read())

model = SentenceTransformer("google/embeddinggemma-300m")

print("Starting encoding...")

embeddings = model.encode([umls_ents[key]['concept_name'] for key in list(umls_map.keys())],
                          batch_size=128, truncate_dim=EMBEDDING_DIM, device='cuda', 
                          show_progress_bar=True)

torch.save(embeddings, f"{DICTS_DIR}/features.pt", pickle_protocol=4)

print("Encoding finished. and features saved!")


