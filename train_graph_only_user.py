import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataloader.dataloader import OnlyAuthorGraphDataLoader, only_author_collate_fn
from src.model.model import OnlyUserGraphTransH
from src.loss.loss import OnlyUserTransXLoss
from tqdm import tqdm
from torch.optim import AdamW
import json

dataset_folder = 'political_science'
dataset_name = 'political_science'
shuffle = True
batch_size = 16384
lr = 1e-3
n_relations = 3
trans_mode = 'transh'
device = 'cpu'
max_epoch = 100


with open(os.path.join(dataset_folder, 'user_id_to_index_to_index.json'), 'r') as f:
    user_id_to_index = json.load(f)

with open(os.path.join('embeddings', dataset_name, 'all_minilm.json'), 'r') as f:
    doc_id_to_index = json.load(f)
    
train_data = DataLoader(
    OnlyAuthorGraphDataLoader(
        f'./{dataset_folder}/author_graph.json',
        user_id_to_index,
        doc_id_to_index
    ),
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=only_author_collate_fn
)

doc_embs = torch.load(f'./embeddings/{dataset_name}/all_minilm.pt').to(device)

model = OnlyUserGraphTransH(
        n_authors=len(user_id_to_index),
        doc_embs=doc_embs,
        n_relations=n_relations,
        normalize=True,
        mode=trans_mode,
        device=device,
    )

loss_fn = OnlyUserTransXLoss(0.5)
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

    torch.save(model.state_dict(), f'models/{dataset_name}/{trans_mode}/only_user.pt')
