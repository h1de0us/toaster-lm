from tqdm import tqdm
import torch
from torch import nn

import wandb


def train(model, optimizer, criterion, train_loader, test_loader, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_losses, test_losses = [], []
    wandb.init()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            src = batch['texts']
            src = src.to(device)
            tgt_out = src[:, 1:]
            src = src[:, :-1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_out.shape[1]).to(device)
            logits = model(src, tgt_mask)
            optimizer.zero_grad()
            
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            running_loss += loss.item() * src.shape[0]


        train_losses += [running_loss / len(train_loader.dataset)]
        wandb.log({"training_loss": train_losses[-1]})
        # print(f'train loss after training epoch {epoch} is {train_losses[-1]}')
        running_loss = 0
        model.eval()
        for batch in test_loader:
            src = batch['texts']
            src = src.to(device)
            tgt_out = src[:, 1:]
            src = src[:, :-1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_out.shape[1]).to(device)
            logits = model(src, tgt_mask)

            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            running_loss += loss.item() * src.shape[0]
        test_losses += [running_loss / len(test_loader.dataset)]
        wandb.log({"test_loss": test_losses[-1]})
        # print(f'val loss after validation epoch {epoch} is {test_losses[-1]}')

    wandb.finish()
    return train_losses, test_losses
