import torch
import torch.nn as nn

class PeepholeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.W_c = nn.Parameter(torch.zeros(3, hidden_size))  # peephole weights: i, f, o

    def forward(self, x, states):
        h, c = states
        gates = self.W_x(x) + self.W_h(h)
        i, f, g, o = gates.chunk(4, 1)
        i = torch.sigmoid(i + c * self.W_c[0])
        f = torch.sigmoid(f + c * self.W_c[1])
        g = torch.tanh(g)
        c_new = f * c + i * g
        o = torch.sigmoid(o + c_new * self.W_c[2])
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(PeepholeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            PeepholeLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        for t in range(seq_len):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, (h[l], c[l]))
                inp = h[l]
            inp = self.dropout(inp)
        out = self.fc(inp)
        return out
