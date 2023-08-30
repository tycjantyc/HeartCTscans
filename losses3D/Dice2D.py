import torch
import torch.nn as nn
import numpy as np

# TODO TEST
class DiceLoss2D(nn.Module):
    def __init__(self, classes, epsilon=1e-5, sigmoid_normalization=True):
        super(DiceLoss2D, self).__init__()
        self.epsilon = epsilon
        self.classes = classes

        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)  # TODO test ?

    def flatten(self, tensor):
        return tensor.view(self.classes, -1)

    def expand_as_one_hot(self, target):
        """
        Converts label image to CxHxW, where each label gets converted to
        its corresponding one-hot vector
        :param target is of shape  (1xHxW)
        :return: 3D output tensor (CxHxW) where C is the classes
        """
        shape = target.size()
        shape = list(shape)
        shape.insert(1, self.classes)
        shape = tuple(shape)

        # expand the input tensor to Nx1xHxW
        src = target.unsqueeze(1)
        
        return torch.zeros(shape).to(target.device).scatter_(1, src, 1).squeeze(0)

    def compute_per_channel_dice(self, input, target):
        epsilon = 1e-5
        target = self.expand_as_one_hot(target.long()).squeeze()

        #import cv2
        #output = target[2]
        #output = output.cpu()
        #output = output.detach().numpy()
        #output = output.reshape(512, 512)
        #output *= 255
        #output = output.astype(np.uint8)
        #cv2.imshow("maska", output)

        if len(target.shape) == 3:
            target = target.reshape(1, 14, 512, 512)
        
        assert input.size() == target.size(), "input' and 'target' must have the same shape"+ str(input.size()) + " and " + str(target.size())

        target = self.flatten(target).float()
        input = self.flatten(input)
        
        # Compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        denominator = (input + target).sum(-1)
        return 2. * intersect / denominator.clamp(min=epsilon)

    def forward(self, input, target):
        input = self.normalization(input)
        per_channel_dice = self.compute_per_channel_dice(input, target)
        DSC = per_channel_dice.clone().cpu().detach().numpy()
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice), DSC
