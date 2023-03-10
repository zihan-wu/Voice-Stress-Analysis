import torch.nn as nn

class CLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, 9, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 256, (4,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.bilstm = nn.LSTM(input_size=3584, hidden_size=512, num_layers=1, bidirectional=True, batch_first=True)
        self.fc_final = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)       # (batch, ch, mel, time)
        x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C * D))  # (batch, time, mel*ch)
        # x = self.fc(x)
        self.bilstm.flatten_parameters()
        x, _ = self.bilstm(x)
        x = self.fc_final(x[:, -1, :])
        return x
    

# if __name__ == '__main__':
#     import torch
#     from torchinfo import summary
#     device = torch.device("cuda") # PyTorch v0.4.0
#     tensor = torch.randn(2, 1, 64, 96).to(device)
#     model = CLSTM().to(device)
#     x = model.forward(tensor)
#     print(x.shape)
    # summary(model, input_size=(2, 1, 64, 96))
