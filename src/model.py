import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F

from src.dataset import TextDataset


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
    
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, embed_dim):
        pass

    def forward(self, x):
        pass

class TransformerDecoder(nn.Module):
    # TODO: add flash attention
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int, # the number of expected features in the input 
                 n_blocks: int,
                 n_head: int,
                 ff_dim: int,
                 text_dataset: TextDataset,
                 use_flash_attention: bool = False,
                 use_rope: bool = False,
                 ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.embeds = nn.Embedding(vocab_size, embed_dim)
        if use_rope:
            self.pos_embeds = RotaryPositionEmbedding(embed_dim)
        else:
            self.pos_embeds = PositionalEncoding(embed_dim)
        self.decoder = nn.ModuleList([nn.TransformerEncoderLayer(embed_dim, n_head, ff_dim, batch_first=True) for _ in range(n_blocks)])
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

        self.dataset = text_dataset


    def _forward(self, x, mask):
        x = self.embeds(x)
        x = self.pos_embeds(x)

        for decoder_block in self.decoder:
            x = decoder_block(x, src_mask=mask, is_causal=True)
        x = self.fc(x)
        return x


    def forward(self, x, mask):
        if self.use_flash_attention:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, 
                enable_math=True, 
                enable_mem_efficient=True
            ):
                return self._forward(x, mask)
        return self._forward(x, mask)
        

    def generate(self, text_dataset, prompt="", max_len=100):
        with torch.no_grad():
            # prompt initialization
            input_ids = text_dataset.text2ids(prompt)[:-1] # remove EOS token
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            # (1, seq_len,)

            mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()

            for _ in range(max_len):
                output = self.forward(input_ids, mask)
                output_probs = F.softmax(output[:, -1, :], dim=-1)
                next_token = torch.multinomial(output_probs, 1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

            generated_text = text_dataset.ids2text(input_ids.squeeze().tolist())
            return generated_text
    

# TODO: generation with beam search
    def generate_with_beam_search(self, text_dataset, prompt="", max_len=100):
        pass
