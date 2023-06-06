import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Initializing the DobleConvolutional layer
class DobleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DobleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

# The code of the model architecture built based on the UNet architecture
class Model(nn.Module):
    def __init__(
            self, in_channels=3, out_chanels=1, features=[64, 128, 256, 512],
    ):
        super(Model, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of Model
        for feature in features:
            self.downs.append(DobleConv(in_channels, feature))
            in_channels = feature

        # Up part of Model
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DobleConv(feature*2, feature))

        self.bottleneck = DobleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_chanels, kernel_size=1)

    # Forward prop
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)