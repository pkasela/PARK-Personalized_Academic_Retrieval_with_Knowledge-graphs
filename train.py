import os

from torch.utils.data import DataLoader
from dataloader import read_jsonl, GraphDataLoader, collate_fn


datafolder = 'computer_science'
split = 'train'

queries    = read_jsonl(os.path.join(datafolder, split, 'queries.jsonl'))
collection = read_jsonl(os.path.join(datafolder, 'collection.jsonl'))
authors    = read_jsonl(os.path.join(datafolder, 'authors.jsonl'))
out_refs   = read_jsonl(os.path.join(datafolder, 'out_refs.jsonl'))
authors_ids = list(set([q['user_id'] for q in queries]))
author_id_to_index = {a_id: i for i, a_id in enumerate(authors_ids)}

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
venue_id_to_index['-1'] = len(venue_ids)

train_data = DataLoader(GraphDataLoader(queries, collection, authors, out_refs), batch_size=4, shuffle=True, collate_fn=collate_fn)

split = 'val'
queries = read_jsonl(os.path.join(datafolder, split, 'queries.jsonl'))
val_data = DataLoader(GraphDataLoader(queries, collection, authors, out_refs), batch_size=4, shuffle=True, collate_fn=collate_fn)


model = GraphBiEncoder(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
    author_to_index=author_id_to_index,
    venue_to_index=venue_id_to_index,
    device='cuda'
)

for d in train_data:
    break