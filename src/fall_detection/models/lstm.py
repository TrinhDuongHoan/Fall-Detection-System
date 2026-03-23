import torch
import torch.nn as nn

class FallLSTM(nn.Module):
    def __init__(self, seq_len=30, feature_dim=66, num_classes=3, hidden_dim1=64, hidden_dim2=128):
        super().__init__()
        
        # Lớp LSTM thứ nhất (Tương đương LSTM(64, return_sequences=True))
        self.lstm1 = nn.LSTM(
            input_size=feature_dim, 
            hidden_size=hidden_dim1, 
            num_layers=1, 
            batch_first=True
        )
        
        # Lớp LSTM thứ hai (Tương đương LSTM(128, return_sequences=False))
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1, 
            hidden_size=hidden_dim2, 
            num_layers=1, 
            batch_first=True
        )
        
        # Bộ phân loại (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        
        # Đi qua khối LSTM 1
        x, _ = self.lstm1(x)
        
        # Đi qua khối LSTM 2
        # h_n shape: (num_layers, batch_size, hidden_size)
        _, (h_n, _) = self.lstm2(x)
        
        # Lấy hidden state cuối cùng (ở time step cuối)
        out = h_n[-1] # shape: (batch_size, hidden_dim2)
        
        # Đưa vào bộ phân loại
        logits = self.classifier(out)
        return logits
