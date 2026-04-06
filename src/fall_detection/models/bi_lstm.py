import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_classes=3, dropout_prob=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True 
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, 128) 
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        x = self.dropout(out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x