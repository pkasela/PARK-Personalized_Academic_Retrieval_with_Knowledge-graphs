import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader.ablation_dataloader import OnlyAuthorGraphDataLoader, only_author_collate_fn
from model.ablation_model import OnlyUserGraphTransX
from loss.ablation_loss import OnlyUserTransXLoss
from tqdm import tqdm
from torch.optim import AdamW
import json
import click

@click.command()
@click.option('--dataset_folder', default='../computer_science', help='Dataset folder')
@click.option('--dataset_name', default='computer_science', help='Dataset name')
@click.option('--model_save_name', default='all_minilm', help='Model name')
@click.option('--shuffle/--no_shuffle', default=True, help='Shuffle data')
@click.option('--batch_size', default=16384, type=int, help='Batch size')
@click.option('--lr', default=1e-3, type=float, help='Learning rate')
@click.option('--n_relations', default=5, type=int, help='Number of relations')
@click.option('--trans_mode', default='transh', help='Trans mode')
@click.option('--device', default='cuda:0', help='Device')
@click.option('--max_epoch', default=100, type=int, help='Max epochs')
@click.option('--embeddings_folder', default='../embeddings', help='Embeddings folder')
@click.option('--model_dir', default='../models', help='Model name')
def main(dataset_folder, dataset_name, model_save_name, shuffle, batch_size, lr, n_relations, trans_mode, device, max_epoch, embeddings_folder, model_dir):

    with open(os.path.join(dataset_folder, 'user_id_to_index_to_index.json'), 'r') as f:
        user_id_to_index = json.load(f)

    with open(os.path.join(embeddings_folder, dataset_name, f'{model_save_name}.json'), 'r') as f:
        doc_id_to_index = json.load(f)
        
    train_data = DataLoader(
        OnlyAuthorGraphDataLoader(
            f'{dataset_folder}/author_graph.json',
            user_id_to_index,
            doc_id_to_index
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=only_author_collate_fn
    )

    doc_embs = torch.load(os.path.join(embeddings_folder, dataset_name, f'{model_save_name}.pt')).to(device)

    model = OnlyUserGraphTransX(
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
    os.makedirs(os.path.join(model_dir, dataset_name, trans_mode), exist_ok=True)
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

        torch.save(model.state_dict(), os.path.join(model_dir, dataset_name, trans_mode, 'only_user.pt'))

if __name__ == '__main__':
    main()