import json
import os
import subprocess
from tqdm import tqdm
from indxr import Indxr


def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def read_jsonl(path, verbose=False):
    with open(path) as f:
        pbar = tqdm(f, total=file_len(path), desc=f'Loading {path}') if verbose else f
        data = [json.loads(line) for line in pbar]
    return data

def write_jsonl(path, data, verbose=True):
    with open(path, 'w') as f:
        pbar = tqdm(data, desc=f'Writing {path}') if verbose else data
        for d in pbar:
            json.dump(d, f)
            f.write('\n')

# remove the authors with less than 20 papers:
dataset_folder = 'computer_science'
output_folder = f'cleaned_{dataset_folder}'
split_year = 2017


val_queries   = Indxr(os.path.join(dataset_folder, 'val/queries.jsonl'))
test_queries  = Indxr(os.path.join(dataset_folder, 'test/queries.jsonl'))
val_rerank_res = set([docs for queries in val_queries for docs in queries['bm25_doc_ids']])
test_rerank_res = set([docs for queries in test_queries for docs in queries['bm25_doc_ids']])

rerank_docs = val_rerank_res.union(test_rerank_res)


authors = Indxr(os.path.join(dataset_folder, 'authors.jsonl'))

val_queries   = Indxr(os.path.join(dataset_folder, 'val/queries.jsonl'))
test_queries  = Indxr(os.path.join(dataset_folder, 'test/queries.jsonl'))
query_authors = set([query['user_id'] for query in tqdm(val_queries)]).union(set([query['user_id'] for query in tqdm(test_queries)]))

final_authors = []
doc_authors = {doc_id: [] for doc_id in list(rerank_docs)}

for a in tqdm(authors):
    user_docs = set(doc['doc_id'] for doc in a['docs'])
    user_docs = user_docs & rerank_docs
    if user_docs or (a['id'] in query_authors):
        final_authors.append({
            'id': a['id'],
            'affiliation': a['affiliation_id'],
            'user_docs': list(doc['doc_id'] for doc in a['docs'])
        })




## entities
# users
# documents
# affiliations
# venues

## relations
# 0 wrote (user wrote doc) ok
# 1 cited (user cited doc) ok
# 2 co_author with (user co_author user) ok
# 3 in_venue (user in_venue venue) ok
# 4 affiliated_to (user affiliated_to affiliation) ok


def get_venue(doc_id):
    doc = collection[doc_id]
    if doc['conference_instance_id'] == "":
        if doc['conference_series_id'] == "":
            venue_id = doc['journal_id']
        else:
            venue_id = doc['conference_series_id']
    else:
        venue_id = doc['conference_instance_id']

    return venue_id


doc_id_to_user = {}

for a in tqdm(final_authors):
    for doc in a['user_docs']:
        if doc in doc_id_to_user:
            doc_id_to_user[doc].append(a['id'])
        else:
            doc_id_to_user[doc] = [a['id']]

out_refs   = read_jsonl(os.path.join(dataset_folder, 'out_refs.jsonl'))
out_refs = {doc['doc_id']: doc for doc in out_refs}
collection = read_jsonl(os.path.join(dataset_folder, 'collection.jsonl'))
collection = {doc['id']: doc for doc in collection}

# final_triplets = []
final_json = {}
for a in tqdm(final_authors):
    final_json[a['id']] = {}
    # wrote
    # final_triplets.extend([
    #     (a['id'], 0, doc)
    #     for doc in a['user_docs']
    # ]
    # )
    final_json[a['id']]['wrote'] = [doc for doc in a['user_docs']]
    # cited
    # final_triplets.extend([
    #     (a['id'], 1, cited_doc)
    #     for doc in a['user_docs']
    #     for cited_doc in out_refs.get(doc)['out_refs']
    # ]
    # )
    final_json[a['id']]['cited'] = [cited_doc for doc in a['user_docs'] for cited_doc in out_refs.get(doc)['out_refs']]
    #co_author
    # final_triplets.extend([
    #     (a['id'], 2, user)
    #     for doc in a['user_docs']
    #     for user in doc_id_to_user.get(doc)
    # ]
    # )
    final_json[a['id']]['co_author'] = [user for doc in a['user_docs'] for user in doc_id_to_user.get(doc)]
    #venue
    # final_triplets.extend([
    #     (a['id'], 3, get_venue(doc))
    #     for doc in a['user_docs']
    #     if get_venue(doc) != ""
    # ]
    # )
    final_json[a['id']]['venue'] = [get_venue(doc) for doc in a['user_docs']]
    # affiliation
    # if a['affiliation'] != "":
    #     final_triplets.extend([(a['id'], 4, a['affiliation'])])
    final_json[a['id']]['affiliation'] = [a['affiliation']]

with open(os.path.join(dataset_folder, 'author_graph.json'), 'w') as f:
    json.dump(final_json, f)

# with open(os.path.join(dataset_folder, 'author_graph.triplets'), 'w') as f:
#     for triplet in tqdm(final_triplets):
#        f.write(triplet[0] + ',' + str(triplet[1]) + ',' + triplet[2] + '\n')


collection = Indxr(os.path.join(dataset_folder, 'collection.jsonl'))

venue_ids = []
for doc in collection:
    if doc['conference_instance_id'] == "":
        if doc['conference_series_id'] == "":
            venue_id = doc['journal_id']
        else:
            venue_id = doc['conference_series_id']
    else:
        venue_id = doc['conference_instance_id']

    venue_ids.append(venue_id)
venue_ids = list(set(venue_ids))
venue_id_to_index = {v_id: i for i, v_id in enumerate(venue_ids)}

with open(os.path.join(dataset_folder, 'venue_id_to_index.json'), 'w') as f:
    json.dump(venue_id_to_index, f)

with open(os.path.join(dataset_folder, 'author_graph.json'), 'r') as f:
    final_json = json.load(f)

affiliation_ids = list(set([item['affiliation'][0] for key, item in final_json.items()]))
affiliation_id_to_index = {a_id: i for i, a_id in enumerate(affiliation_ids)}

with open(os.path.join(dataset_folder, 'affiliation_id_to_index.json'), 'w') as f:
    json.dump(affiliation_id_to_index, f)


user_ids = list(set([key for key, item in final_json.items()]))
user_id_to_index = {u_id: i for i, u_id in enumerate(user_ids)}

with open(os.path.join(dataset_folder, 'user_id_to_index_to_index.json'), 'w') as f:
    json.dump(user_id_to_index, f)