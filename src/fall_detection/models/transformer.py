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
        self.d_model = d_model
        
        # 1. Temporal Convolution (Captures Joint Velocity between consecutive frames)
        self.temporal_proj = nn.Sequential(
            nn.Conv1d(feature_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # 2. Learnable CLS Token (Giống hệ tư tưởng của Vision Transformers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Positional Embedding (seq_len + 1 để chứa thêm token CLS)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # 4. Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Classifier Head
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
        
        # 1. 1D Convolution mapping
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.temporal_proj(x)
        # Back to (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        
        # 2. Append CLS Token at the beginning of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add Positional Embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # 4. Forward through Transformer
        x = self.transformer_encoder(x)
        
        # 5. Extract CLS token output (index 0) instead of Global Average Pooling
        # Token này đã gom cụm trạng thái chớp nhoáng (ví dụ như cú ngã)
        cls_output = x[:, 0, :]
        
        # 6. Classify
        logits = self.classifier(cls_output)
        return logits
