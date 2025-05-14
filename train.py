import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from src.dataloader.dataloader import read_jsonl, GraphDataLoader, collate_fn
from src.model.model import GraphBiEncoder
from src.loss.loss import TripletMarginLoss
from tqdm import tqdm
from torch.optim import AdamW


datafolder = 'computer_science'
split = 'train'
shuffle = True
batch_size = 256
lr = 5e-5

train_queries    = read_jsonl(os.path.join(datafolder, split, 'queries.jsonl'))
collection = read_jsonl(os.path.join(datafolder, 'collection.jsonl'))
authors    = read_jsonl(os.path.join(datafolder, 'authors.jsonl'))
out_refs   = read_jsonl(os.path.join(datafolder, 'out_refs.jsonl'))
authors_ids = list(set([q['user_id'] for q in train_queries]))
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

train_data = DataLoader(GraphDataLoader(train_queries, collection, authors, out_refs), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

split = 'val'
val_queries = read_jsonl(os.path.join(datafolder, split, 'queries.jsonl'))
val_data = DataLoader(GraphDataLoader(val_queries, collection, authors, out_refs), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


model = GraphBiEncoder(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
    author_to_index=author_id_to_index,
    venue_to_index=venue_id_to_index,
    doc_embeddings='embeddings/all_minilm.pt',
    doc_id_to_index='embeddings/all_minilm.json',
    query_embeddings='embeddings/all_minilm_query.pt',
    query_id_to_index='embeddings/all_minilm_query.json',
    device='cuda'
)
model.load(f'../multidomain/models/{datafolder}/all_minilm.pt')

loss_fn = TripletMarginLoss(0.1)
epoch = 0
optimizer = AdamW(model.parameters(), lr=lr)
best_loss = 999
for epoch in tqdm(range(50)):
    losses = []
    accuracies = []
    pbar = tqdm(train_data)
    for d in pbar:
        optimizer.zero_grad()
        output = model(d)
        loss_val, accuracy = loss_fn(output)

        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss_val.cpu().detach().item())
        accuracies.extend(accuracy.tolist())

        average_loss = np.mean(losses)
        average_sim_accuracy = np.mean(accuracies)

        summary = "TRAIN EPOCH {:3d} Sim Accuracy {}, Average Loss {:.2e}".format(epoch, round(average_sim_accuracy*100,2), average_loss)
        pbar.set_description(summary)

    losses = []
    accuracies = []    
    pbar = tqdm(val_data)
    for d in pbar:
        with torch.no_grad():
            output = model(d)
            loss_val, accuracy = loss_fn(output)

        losses.append(loss_val.cpu().detach().item())
        accuracies.extend(accuracy.tolist())

        average_loss = np.mean(losses)
        average_sim_accuracy = np.mean(accuracies)

        summary = "VAL EPOCH {:5d} Sim Accuracy {}, Average Loss {:.2e}".format(epoch, round(average_sim_accuracy*100,2), average_loss)
        pbar.set_description(summary)

    if average_loss < best_loss:
        best_loss = average_loss
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/trained_model.pt')