import torch
from torch import nn
from torch.utils.data import Dataset
from torch import functional as F

from dataset import TextDataset


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position / torch.pow(10000, torch.arange(0, embed_dim, 2).float() / embed_dim))
        pe[:, 1::2] = torch.cos(position / torch.pow(10000, torch.arange(0, embed_dim, 2).float() / embed_dim))
        pe = pe.unsqueeze(0)
        # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int, # the number of expected features in the input 
                 n_blocks: int,
                 n_head: int,
                 ff_dim: int,
                 text_dataset: TextDataset
                 ) -> None:
        super().__init__()
        self.embeds = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeds = PositionalEncoding(embed_dim)
        self.decoder = nn.ModuleList([nn.TransformerEncoderLayer(embed_dim, n_head, ff_dim, batch_first=True) for _ in range(n_blocks)])
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

        self.dataset = text_dataset

    def forward(self, x, mask):
        # x.shape is (batch_size, seq_len)
        # mask.shape is (seq_len, seq_len)
        x = self.embeds(x)
        # print(x.shape)
        x = self.pos_embeds(x)
        # print(x.shape)
        for decoder_block in self.decoder:
            x = decoder_block(x, mask)
            # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x


    def generate(self, prompt="", max_len=100):
        # prompt initialization
        input_ids = self.text_dataset.text2ids(prompt)
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        memory = self.decoder.embedding(input_ids)
        memory = self.decoder.positional_encoding(memory)

        tgt_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()

        for _ in range(max_len):
            output = self.decoder(input_ids, memory, tgt_mask)
            output_probs = F.softmax(output[:, -1, :], dim=-1)
            next_token = torch.multinomial(output_probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)

        generated_text = self.text_dataset.ids2text(input_ids.squeeze().tolist())
        return generated_text
