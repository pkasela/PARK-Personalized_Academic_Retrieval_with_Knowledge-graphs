import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataloader.dataloader import UserVenueAuthorGraphDataLoader, user_venue_author_collate_fn
from src.model.model import UserVenueGraphTransH
from src.loss.loss import UserVenueTransXLoss
from tqdm import tqdm
from torch.optim import AdamW
import json

dataset_folder = 'psychology'
dataset_name = 'psychology'
shuffle = True
batch_size = 16384
lr = 1e-3
n_relations = 4
trans_mode = 'transh'
device = 'cuda'
max_epoch = 100


with open(os.path.join(dataset_folder, 'user_id_to_index_to_index.json'), 'r') as f:
    user_id_to_index = json.load(f)

with open(os.path.join(dataset_folder, 'venue_id_to_index.json'), 'r') as f:
    venue_id_to_index = json.load(f)

with open(os.path.join('embeddings', dataset_name, 'all_minilm.json'), 'r') as f:
    doc_id_to_index = json.load(f)
    
train_data = DataLoader(
    UserVenueAuthorGraphDataLoader(
        f'./{dataset_folder}/author_graph.json',
        user_id_to_index,
        venue_id_to_index,
        doc_id_to_index
    ),
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=user_venue_author_collate_fn
)

doc_embs = torch.load(f'./embeddings/{dataset_name}/all_minilm.pt').to(device)

model = UserVenueGraphTransH(
        n_authors=len(user_id_to_index),
        n_venues=len(venue_id_to_index),
        doc_embs=doc_embs,
        venue_pad_id=venue_id_to_index[''],
        n_relations=n_relations,
        normalize=True,
        mode=trans_mode,
        device=device,
    )

loss_fn = UserVenueTransXLoss(0.5)
optimizer = AdamW(model.parameters(), lr=lr)
optimizer.zero_grad()
print(dataset_name)
os.makedirs(f'models/{dataset_name}/{trans_mode}', exist_ok=True)
for epoch in tqdm(range(max_epoch)):
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

    torch.save(model.state_dict(), f'models/{dataset_name}/{trans_mode}//user_venue.pt')
