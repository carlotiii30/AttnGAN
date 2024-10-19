import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_channels, text_dim, ndf):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(image_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 2, 1, bias=False),
        )
        self.text_proj = nn.Linear(text_dim, ndf * 2)

    def forward(self, image, text_embedding):
        x = self.conv_blocks(image)
        text_proj = self.text_proj(text_embedding).unsqueeze(2).unsqueeze(3)
        return torch.sigmoid(x + text_proj)
