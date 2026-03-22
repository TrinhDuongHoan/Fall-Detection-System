import torch
import torch.nn as nn

class FallTransformer(nn.Module):
    def __init__(
        self, 
        seq_len=30, 
        feature_dim=66, 
        num_classes=3, 
        d_model=128, 
        num_heads=4, 
        ff_dim=256, 
        num_layers=3, 
        dropout=0.3
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, feature_dim)
        return: (batch_size, num_classes) # Raw logits for CrossEntropyLoss
        """
        batch_size = x.size(0)

        x = self.feature_proj(x)
        
        positions = torch.arange(0, self.seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, self.seq_len)
        
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        
        logits = self.classifier(x)
        return logits
