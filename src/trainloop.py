from tqdm import tqdm
import torch
from torch import nn
from torch import autocast
import wandb

from src.utils import save_checkpoint, get_grad_norm


def train(model, 
          optimizer, 
          criterion, 
          train_loader, 
          test_loader, 
          num_epochs, 
          scheduler=None, 
          clip_grad=False,
          max_gradient_norm=1., 
          log_epoch=100):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_losses, test_losses = [], []
    wandb.init()


    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            src = batch['texts']
            src = src.to(device)
            tgt_out = src[:, 1:]
            src = src[:, :-1]
            mask = nn.Transformer.generate_square_subsequent_mask(tgt_out.shape[1]).to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(src, mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            
            loss.backward() # we do not scale the loss to not have issues with precision
            running_loss += loss.item() * src.shape[0]

            scaler.scale(loss)

            # Gradient Clipping
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            scaler.unscale_(optimizer)

            if clip_grad:
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

            if scheduler is not None:
                wandb.log({"lr": scheduler.get_last_lr()[0]})
                scheduler.step()
            else:
                wandb.log({"lr": optimizer.param_groups[0]['lr']})

            # logging grad norm to wandb
            wandb.log({"grad_norm": get_grad_norm(model)})
                
            # Updates the scale for next iteration.
            scaler.update()


        train_losses += [running_loss / len(train_loader.dataset)]
        wandb.log({"training_loss": train_losses[-1]})
        # print(f'train loss after training epoch {epoch} is {train_losses[-1]}')
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                src = batch['texts']
                src = src.to(device)
                tgt_out = src[:, 1:]
                src = src[:, :-1]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_out.shape[1]).to(device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(src, tgt_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                running_loss += loss.item() * src.shape[0]
            test_losses += [running_loss / len(test_loader.dataset)]
            wandb.log({"test_loss": test_losses[-1]})
        # print(f'val loss after validation epoch {epoch} is {test_losses[-1]}')


        if epoch % log_epoch == 0:
            save_checkpoint(checkpoint_dir="checkpoints",
                            model=model, 
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch, 
                            save_best=False, 
                            only_best=False)
            

    wandb.finish()
    return train_losses, test_losses
