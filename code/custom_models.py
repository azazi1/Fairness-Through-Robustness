# Custom model implementations
# #### Some rules:
# 1. Each model's final layer would be called fc3
# 2. For CNN based models, the model will typically have some convolution part and some fully connected part,
# each model class should define self.model_conv and self.model_linear which will define the two parts respectively
# 3. Doing these 2 simple things wouldn't break rest of the code

import torch.nn as nn

class MLP1(nn.Module):
    def __init__(self, inputs, num_classes):
        super(MLP1, self).__init__()
        self.n_in = inputs
        self.fc3 = nn.Linear(self.n_in, num_classes)

    def forward(self, x):
        x = x.reshape(-1, self.n_in)
        out = self.fc3(x)
        return out


class UTKClassifier(nn.Module):
    def __init__(self, num_classes):
        super(UTKClassifier, self).__init__()
        layers_conv = [nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=2),
                       nn.ReLU(),
                       nn.Dropout(0.3),
                       nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                       nn.ReLU(),
                       nn.Dropout(0.3),
                       nn.MaxPool2d(kernel_size=2, stride=2)]

        self.model_conv = nn.Sequential(*layers_conv)
        layers_linear = [nn.Linear(153664, 256), # This looks bad
                         nn.ReLU(),
                         nn.Dropout(0.5)]
        self.model_linear = nn.Sequential(*layers_linear)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        out = self.model_conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.model_linear(out)
        out = self.fc3(out)
        return out