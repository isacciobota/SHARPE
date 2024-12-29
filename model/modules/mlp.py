import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.fc_1 = nn.Linear(in_features, hidden_features)
        self.fc_2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc_2(x)
        x = self.drop(x)
        return x
