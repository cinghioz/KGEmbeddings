import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import os
import pickle

os.chdir("/home/cc/phd/KGEmbeddings")

# PATH = "/home/marco_dossena/PHD/KGEmbeddings/"
PATH = "/home/cc/phd/KGEmbeddings/"
EMBEDDING_DIM = 512
DATA = "umls"
MODEL_NAME = "TransE"
# MODEL_PATH = "/home/cc/phd/KGEmbeddings/models/TransE_FB15k_0/"
# MODEL_PATH = "/home/cc/phd/KGEmbeddings/models/RotatE_FB15k_0/"
MODEL_PATH = f"{PATH}models/{MODEL_NAME}_{DATA}_0"
# DICTS_DIR = "/home/cc/phd/KGEmbeddings/data/FB15k/"
DICTS_DIR = f"{PATH}data/{DATA}"

e_map = pd.read_json("/home/cc/phd/KGEmbeddings/data/umls/entity_map.json", typ='series').to_dict()
r_map = pd.read_json("/home/cc/phd/KGEmbeddings/data/umls/rel_map.json", typ='series').to_dict()

inv_e_map = {v: k for k, v in e_map.items()}
inv_r_map = {v: k for k, v in r_map.items()}

def api_call(cui):
    url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}?apiKey=f72ff16d-f1da-40a6-adbc-9f42ff7c9fe7"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('result', {}).get('name', 'N/A')
    else:
        return 'N/A'

if __name__ == "__main__":

    umls = pd.read_csv("/home/cc/phd/KGEmbeddings/data/umls/train.csv", low_memory=False)
    umls_r5 = umls[umls['relation_id'] == 5]

    number_of_proj = 2
    shared_tails = umls_r5.groupby('tail_id')['head_id'].nunique()
    shared_tails = shared_tails[shared_tails > (number_of_proj-1)]  # tails with more than number_of_proj heads

    shared_tails = shared_tails.sample(frac=1, random_state=77)  # shufflle series

    queries = [] 
    shared_tails_inter = []
    results = []  

    if shared_tails.empty:
        print("No shared tails found with relation_id = 5")
    else:
        for shared_tail_id in tqdm(shared_tails.index, desc="Processing queries"):
            # Get the heads pointing to this shared tail
            heads = umls_r5[umls_r5['tail_id'] == shared_tail_id]['head_id'].unique()[:number_of_proj]
            if len(heads) < number_of_proj:
                continue  # need at least number_of_proj heads

            relations = umls_r5[umls_r5['tail_id'] == shared_tail_id]['relation_id'].values

            # Save query structure (5 is the relation_id for "is_associated_with")
            query = [[(heads[i], relations[i]) for i in range(number_of_proj)]]

            # new head = shared_tail_id
            new_head_id = shared_tail_id
            new_edges = umls[(umls['head_id'] == new_head_id) & (umls['relation_id'] != 0)]

            if new_edges.empty:
                continue
            
            # Group tails by relations
            relation_dict = (
                new_edges.groupby('relation_id')['tail_id']
                .apply(list)
                .to_dict()
            )
            
            for rel, tails in relation_dict.items():
                queries.append(query+[rel])
                shared_tails_inter.append(shared_tail_id)
                results.append(tails)

    string_queries = []

    template = """{sh_tail} is associated with {h1} and {h2}. What other entities are related to {sh_tail} through the relation "{target_rel}"? """

    for query, sh_tail, res in tqdm(zip(queries, shared_tails_inter, results), total=len(queries), desc="Formatting and save string queries"):

        sh_tail = api_call(inv_e_map[sh_tail])
        h1 = api_call(inv_e_map[query[0][0][0]])
        h2 = api_call(inv_e_map[query[0][1][0]])
        target_rel = "is associated to" if query[1] == 5 else inv_r_map[query[1]]

        query_to_string = template.format(
            sh_tail=sh_tail,
            h1=h1,
            h2=h2,
            target_rel=target_rel
        )

        string_queries.append([query_to_string, [h1, h2], sh_tail, target_rel, [api_call(inv_e_map[r]) for r in res]])

    queries_df = pd.DataFrame(string_queries, columns=["query", "known_heads", "shared_tail", "target_relation", "answers"])
    queries_df.to_csv("umls_generated_queries.csv", index=False)

    save_dict = {
        'queries': queries,
        'results': results
    }

    with open(f'/home/cc/phd/KGEmbeddings/queries/{DATA}/queries-isa.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

