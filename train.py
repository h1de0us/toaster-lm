import torch
import wandb
from torch.utils.data import DataLoader

from src.model import TransformerDecoder
from src.dataset import TextDataset
from src.trainloop import train
from src.utils import collate_fn


if __name__ == "__main__":
    # some hyperparameters
    VOCAB_SIZE = 3000
    MAX_LENGTH = 512
    BATCH_SIZE = 32
    NUM_EPOCHS = 600

    text_dataset = TextDataset(data_file="stories.txt",
                            train=True,
                            sp_model_prefix="sp_model",
                            vocab_size=VOCAB_SIZE,
                            max_length=MAX_LENGTH)


    train_set = TextDataset(data_file="large_train.txt", 
                            train=True, 
                            sp_model_prefix="sp_model", 
                            vocab_size=VOCAB_SIZE, 
                            max_length=MAX_LENGTH)
    valid_set = TextDataset(data_file="large_train.txt", 
                            train=True, 
                            sp_model_prefix="sp_model", 
                            vocab_size=VOCAB_SIZE,
                            max_length=MAX_LENGTH)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)


    wandb.login()
    wandb.init(project="LLM-Homework")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerDecoder(vocab_size=text_dataset.vocab_size,
                                embed_dim=64,
                                n_blocks=2,
                                n_head=2,
                                ff_dim=64,
                                text_dataset=text_dataset,)
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=text_dataset.pad_id)

    train_losses, test_losses = train(model=model, optimizer=optimizer, criterion=criterion,
                                    train_loader=train_loader, test_loader=val_loader, num_epochs=NUM_EPOCHS)
    

    print(model.generate(text_dataset, prompt="Once upon a time there was a", max_len=100))



