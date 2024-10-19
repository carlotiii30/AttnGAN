import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, z_dim, text_dim, image_channels, ngf):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim + text_dim, ngf * 8 * 4 * 4)
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, text_embedding):
        input_vec = torch.cat((noise, text_embedding), dim=1)
        x = self.fc(input_vec).view(-1, 512, 4, 4)
        return self.conv_blocks(x)
