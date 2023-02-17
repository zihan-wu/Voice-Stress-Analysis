import torch.nn as nn
import torch
# from .utils import NetworkCommonMixIn

class AudioNTT2020Task6(nn.Module):
    """DCASE2020 Task6 NTT Solution Audio Embedding Network."""
    def __init__(self, n_mels, d):
        super().__init__()
        # self.features = nn.Sequential(
        #     nn.Conv1d(1, 64, 3, stride=6, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2, stride=2),

        #     nn.Conv1d(64, 64, 3, stride=6, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2, stride=2),

        #     nn.Conv1d(64, 64, 3, stride=6, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2, stride=2),

        #     # nn.Conv1d(64, 64, 6, stride=1, padding=1),
        #     # nn.BatchNorm1d(64),
        #     # nn.ReLU(),
        #     # nn.MaxPool1d(2, stride=2),

        #     # nn.Conv1d(64, 64, 6, stride=1, padding=1),
        #     # nn.BatchNorm1d(64),
        #     # nn.ReLU(),
        #     # nn.MaxPool1d(2, stride=2),

        #     # nn.Conv1d(64, 64, 6, stride=1, padding=1),
        #     # nn.BatchNorm1d(64),
        #     # nn.ReLU(),
        #     # nn.MaxPool1d(2, stride=2),
        # )
        self.fc = nn.Sequential(
            nn.Linear(6373, 3000),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(3000, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x):
        # x = self.features(x)       
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        # print(x.shape)
        D, B, T = x.shape
        x = x.reshape((D, B*T))
        x = self.fc(x)
        return x


class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.

    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    def __init__(self, n_mels=64, d=2048):
        super().__init__(n_mels=n_mels, d=d)

    def forward(self, x):
        x = super().forward(x)
        x = x.mean(0) + x.amax(0)
        x = x.reshape((1,-1))
        assert x.shape[1] == self.d and x.ndim == 2
        # print(x.size())
        return x
    

if __name__ == '__main__':
    import opensmile
    import torch
    from torchsummary import summary
    audio_file = '/home/gelbanna/hdd/vsa-data/CogLoadv2/raw/Voice Stress Project Delivery 2.0/Female/Amanda/Voice with cognitive load_1.wav'
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals)
    smile_vector = smile.process_file(audio_file).to_numpy().flatten()
    smile_torch = torch.from_numpy(smile_vector)
    smile_torch = torch.reshape(smile_torch, (1,1,6373))
    print(smile_torch.size())
    device = torch.device("cpu") # PyTorch v0.4.0
    model = AudioNTT2020().to(device)
    x = model.forward(smile_torch)
    # summary(model, (1,6373))
