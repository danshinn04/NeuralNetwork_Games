import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout = 0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None 
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.encoder_embeddings = nn.Embedding(vocab_size, d_model)
        self.decoder_embeddings = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder_embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        src = self.encoder_embeddings(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        # Encode target sequence
        tgt = self.decoder_embeddings(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        # Pass through transformer
        output = self.transformer(src, tgt)
        # Pass output through linear layer for final predictions
        output = self.decoder(output)
        return output
    
    