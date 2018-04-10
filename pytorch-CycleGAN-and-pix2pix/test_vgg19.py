import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# import utils
# import argparse

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

# Model


class VGGnet(nn.Module):
    def __init__(self):
        super(VGGnet, self).__init__()
        params_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        # Conv 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1.weight.data = torch.FloatTensor(params_dict['conv1_1'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv1_1.bias.data = torch.FloatTensor(params_dict['conv1_1'][1])

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2.weight.data = torch.FloatTensor(params_dict['conv1_2'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv1_2.bias.data = torch.FloatTensor(params_dict['conv1_2'][1])

        self.pool1 = nn.MaxPool2d(2, stride=2)

        # Conv 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_1.weight.data = torch.FloatTensor(params_dict['conv2_1'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv2_1.bias.data = torch.FloatTensor(params_dict['conv2_1'][1])

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2.weight.data = torch.FloatTensor(params_dict['conv2_2'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv2_2.bias.data = torch.FloatTensor(params_dict['conv2_2'][1])

        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Conv 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_1.weight.data = torch.FloatTensor(params_dict['conv3_1'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv3_1.bias.data = torch.FloatTensor(params_dict['conv3_1'][1])

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2.weight.data = torch.FloatTensor(params_dict['conv3_2'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv3_2.bias.data = torch.FloatTensor(params_dict['conv3_2'][1])

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))

        return x


# model = VGGnet()
# model.cuda()
# model.eval()

#image = utils.load_image(args.test)
#image = image[:, :, ::-1]
#VGG_MEAN = np.array([103.939, 116.779, 123.68])
#image = (image * 255.0) - VGG_MEAN
#image = image.transpose(2, 0, 1)
#image = image.astype(np.float32)
#input = torch.from_numpy(image)
#input = input.cuda()
#input_var = torch.autograd.Variable(input, volatile=True)

# output = model(input_var.unsqueeze(0))
#output = output.data.cpu().numpy()
#out = torch.autograd.Variable(torch.from_numpy(output))
#utils.print_prob(F.softmax(out).data.numpy()[0], './synset.txt')
