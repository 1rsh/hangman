import torch
import torch.nn as nn
import math

class Hangformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, max_length, eval=False):
        super(Hangformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.make_positional_encoding(max_length, embedding_dim)
        
        self.guessed_transform = nn.Linear(26, 62)  
        
        total_dim = embedding_dim + 62 + 6  
        encoder_layer = nn.TransformerEncoderLayer(d_model=total_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.inter_fc = nn.Linear(total_dim, 64)
        self.final_fc = nn.Linear(64, 26)
        
    def make_positional_encoding(self, max_length, embedding_dim):
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        self.positional_encoding = torch.zeros(1, max_length, embedding_dim)
        self.positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, word_state, guessed_letters, remaining_trials):
        x = self.embedding(word_state) # (batch_size, max_length, embedding) -> (batch_size, 30, 100)

        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device) # (batch_size, max_length, embedding) -> (batch_size, 30, 100)

        guessed_transformed = self.guessed_transform(guessed_letters.float())  # (batch_size, 62)
        guessed_expanded = guessed_transformed.unsqueeze(1).expand(-1, x.size(1), -1) # (batch_size, max_length, 62) -> (batch_size, 30, 62)
        
        remaining_trials_expanded = remaining_trials.unsqueeze(1).expand(-1, x.size(1), -1) # (batch_size, max_length, max_tries) -> (batch_size, 30, 6)
        
        combined = torch.cat((x, guessed_expanded, remaining_trials_expanded), dim=-1) # (batch_size, max_length, total_dim) -> (batch_size, 30, 168)
        
        encoder_output = self.transformer_encoder(combined) # (batch_size, max_length, total_dim) -> (batch_size, 30, 168)
        
        pooled = encoder_output.mean(dim=1) # (batch_size, total_dim) -> (batch_size, 168)
        
        x = self.inter_fc(pooled) # (batch_size, 64)
        
        logits = self.final_fc(x) # (batch_size, 26)
        
        return logits
    
    def load(self, filepath, device):
        return torch.load(filepath, map_location=device)


