import torch
from typing import List

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_sequence(batch):
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch


def collate_fn(dataset_items: List[dict]):
    return {
        "texts": torch.as_tensor(pad_sequence([item[0] for item in dataset_items])),
        "pad_masks": torch.as_tensor(pad_sequence([torch.ones(item[1]) for item in dataset_items])),
    }


# via https://github.com/WrathOfGrapes/asr_project_template/blob/hw_asr_2022/hw_asr/base/base_trainer.py
def save_checkpoint(checkpoint_dir,
                    model, 
                    optimizer,
                    scheduler,
                    epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        filename = str(checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")


# via https://github.com/WrathOfGrapes/asr_project_template/blob/hw_asr_2022/hw_asr/base/base_trainer.py
@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()
