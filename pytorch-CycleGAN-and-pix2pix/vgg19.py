import torch.nn as nn
from torchvision import models

vgg19_npy_path = './vgg19.npy'
original_model = models.vgg19(pretrained=True)


class VGGnet19(nn.Module):
    def __init__(self):
        super(VGGnet19, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(original_model.features.children())[:14]
        )

    def forward(self, x):
        x = self.features(x)
        return x
