import torch
import torch.nn as nn

class FallBiLSTM(nn.Module):
    def __init__(self, seq_len=30, feature_dim=66, num_classes=3, hidden_dim1=64, hidden_dim2=128):
        super().__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=feature_dim, 
            hidden_size=hidden_dim1, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1 * 2, 
            hidden_size=hidden_dim2, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x): 
        x, _ = self.lstm1(x)
        
        _, (h_n, _) = self.lstm2(x)
        
        hidden_forward = h_n[0] 
        hidden_backward = h_n[1] 
        
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1) 
        
        logits = self.classifier(final_hidden)
        return logits
