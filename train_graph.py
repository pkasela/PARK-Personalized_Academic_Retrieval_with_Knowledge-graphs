import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import AuthorGraphDataLoader, author_collate_fn
from model import GraphTransH
from loss import TransXLoss
from tqdm import tqdm
from torch.optim import AdamW
import json

dataset_folder = 'computer_science'
split = 'train'
shuffle = True
batch_size = 16384
lr = 1e-3


with open(os.path.join(dataset_folder, 'user_id_to_index_to_index.json'), 'r') as f:
    user_id_to_index = json.load(f)

with open(os.path.join(dataset_folder, 'affiliation_id_to_index.json'), 'r') as f:
    affiliation_id_to_index = json.load(f)

with open(os.path.join(dataset_folder, 'venue_id_to_index.json'), 'r') as f:
    venue_id_to_index = json.load(f)

with open(os.path.join('embeddings', 'all_minilm.json'), 'r') as f:
    doc_id_to_index = json.load(f)
    
train_data = DataLoader(
    AuthorGraphDataLoader(
        f'./{dataset_folder}/author_graph.json',
        user_id_to_index,
        affiliation_id_to_index,
        venue_id_to_index,
        doc_id_to_index
    ),
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=author_collate_fn
)

n_relations = 5
doc_embs = torch.load('./embeddings/all_minilm.pt').to('cuda')

model = GraphTransH(
        n_authors=len(user_id_to_index),
        n_venues=len(venue_id_to_index),
        n_affiliations=len(affiliation_id_to_index),
        doc_embs=doc_embs,
        venue_pad_id=venue_id_to_index[''],
        affiliation_pad_id=venue_id_to_index[''],
        n_relations=n_relations,
        mode='transe',
        device='cuda',
    )

loss_fn = TransXLoss(0.5)
optimizer = AdamW(model.parameters(), lr=lr)
optimizer.zero_grad()
for epoch in tqdm(range(100)):
    pbar = tqdm(train_data)
    losses = []
    for data in pbar:
        output = model(data)
        
        loss_val = loss_fn(output)
        
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss_val.cpu().detach().item())
        average_loss = np.mean(losses)
        summary = "TRAIN EPOCH {:3d} Average Loss {:.2e}".format(epoch,  average_loss)
        pbar.set_description(summary)

torch.save(model.state_dict(), 'models/user.pt')
