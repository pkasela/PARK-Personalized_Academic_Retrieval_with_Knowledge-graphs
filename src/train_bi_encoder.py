import os

import numpy as np

from torch import autocast, no_grad
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import MultiDomainDataset
from model.utils import seed_everything
from model.bi_encoder_model import BiEncoder
from loss.bi_encoder_loss import TripletMarginLoss
import click

def train(
        train_dataloader, 
        model, 
        optimizer, 
        loss_fn, 
        device,
        epoch
    ) -> float:

    losses = []
    accuracies = []

    optimizer.zero_grad()

    pbar = tqdm(train_dataloader)
    for batch in pbar:
        with autocast(device_type=device):
            output = model(batch)
        if 't5' in model.model_name:
            loss_val = output.loss
            accuracy = output['accuracy']
        else:
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
    
    return average_loss

def validate(
        train_dataloader, 
        model,  
        loss_fn, 
        device,
        epoch
    ) -> float:

    losses = []
    accuracies = []

    pbar = tqdm(train_dataloader, colour="red")
    for batch in pbar:
        with autocast(device_type=device):
            with no_grad():
                output = model(batch)
        if 't5' in model.model_name:
            loss_val = output.loss
            accuracy = output['accuracy']
        else:
            loss_val, accuracy = loss_fn(output)

        losses.append(loss_val.cpu().detach().item())
        accuracies.extend(accuracy.tolist())
        
        average_loss = np.mean(losses)
        average_sim_accuracy = np.mean(accuracies)

        summary = "VAL EPOCH {:5d} Sim Accuracy {}, Average Loss {:.2e}".format(epoch, round(average_sim_accuracy*100,2), average_loss)
        pbar.set_description(summary)
    
    return average_loss

@click.command()
@click.option('--dataset_folder', default='physics', help='Dataset folder')
@click.option('--dataset_name', default='physics', help='Dataset name')
@click.option('--device', default='cpu', help='Device')
@click.option('--model_dir', default='../models', help='Model name')
@click.option('--model_name', default='all_minilm', help='Model name')
@click.option('--model_save_name', default='all_minilm', help='Model save name')
@click.option('--tokenizer_name', default='all_minilm', help='tokenizer name')
@click.option('--batch_size', default=512, type=int, help='Batch size')
@click.option('--shuffle/--no_shuffle', default=True, help='Shuffle data')
@click.option('--seed', default=42, type=int, help='Seed')
@click.option('--max_tokens', default=512, type=int, help='Max tokens')
@click.option('--normalize/--no_normalize', default=True, help='Normalize')
@click.option('--pooling_mode', default='cls', help='Pooling mode')
@click.option('--lr', default=1e-3, type=float, help='Learning rate')
@click.option('--epoch', default=100, type=int, help='Epochs')
@click.option('--loss_gamma', default=0.5, type=float, help='Loss gamma')
def main(dataset_folder, dataset_name, device, model_dir, model_name, model_save_name, tokenizer_name, batch_size, shuffle, seed, max_tokens, normalize, pooling_mode, lr, epoch, loss_gamma) -> None:

    os.makedirs(os.path.join(model_dir, dataset_name), exist_ok=True)
    
    seed_everything(seed)

    train_dataset = MultiDomainDataset(dataset_folder, split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    val_dataset = MultiDomainDataset(dataset_folder, split='val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    model = BiEncoder(
        model_name,
        tokenizer_name,
        max_tokens,
        normalize,
        pooling_mode,
        device
    )
    model.save(os.path.join(os.path.join(model_dir, dataset_name), f'{model_save_name}.pt'))
    loss_fn = TripletMarginLoss(loss_gamma) 
    optimizer = AdamW(model.parameters(), lr=lr)
    best_val_loss = validate(val_dataloader, model, loss_fn, device, -1)
    for epoch in tqdm(range(epoch)):
        model.train()
        train_loss = train(train_dataloader, model, optimizer, loss_fn, device, epoch + 1)
        model.eval()
        val_loss = validate(val_dataloader, model, loss_fn, device, epoch + 1)

        if val_loss < best_val_loss:
            print(f'Found new best model on epoch: {epoch + 1}, new best validation loss {val_loss}')
            best_val_loss = val_loss

        model.save(os.path.join(os.path.join(model_dir, dataset_name), f'{model_save_name}.pt'))
        

    
if __name__ == '__main__':
    main()