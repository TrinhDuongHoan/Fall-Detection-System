import torch
import torch.nn as nn

class FallHybrid(nn.Module):
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
        self.d_model = d_model
        
        self.temporal_proj = nn.Sequential(
            nn.Conv1d(feature_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes)
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, feature_dim)
        """
        batch_size = x.size(0)
        
        x = x.transpose(1, 2)
        x = self.temporal_proj(x)
        x = x.transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer_encoder(x)
        
        cls_output = x[:, 0, :]
        
        logits = self.classifier(cls_output)
        return logits
