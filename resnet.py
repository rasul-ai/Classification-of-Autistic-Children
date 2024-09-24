import torch.nn as nn
from modules import resnet50

class pretrained_model(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(pretrained_model, self).__init__()
        self.model = resnet50(pretrained=True)

        self.model_fc = nn.Sequential(
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        feature = self.model(x)
        feature = feature.view(feature.size(0), -1)
        output = self.model_fc(feature)

        return output
