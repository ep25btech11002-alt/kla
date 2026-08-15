import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class UNetSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            ResidualBlock(base_features)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_features, base_features*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_features*2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_features*2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_features*2, base_features*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_features*4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_features*4)
        )
        
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_features*4),
            ResidualBlock(base_features*4)
        )
        
        # Decoder upsampling steps
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_features*4, base_features*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features*2),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_features*2, base_features, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(base_features, base_features, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        
        # Refinement after each upsampling (optional)
        self.ref2 = ResidualBlock(base_features*2)
        self.ref1 = ResidualBlock(base_features)
        self.ref0 = ResidualBlock(base_features)
        
        self.final = nn.Conv2d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)          # 128
        enc2 = self.enc2(enc1)       # 64
        enc3 = self.enc3(enc2)       # 32
        
        # Bottleneck
        b = self.bottleneck(enc3)    # 32
        
        # Decoder
        up2 = self.up2(b)            # 64
        up2 = self.ref2(up2 + enc2)  # add skip and refine
        
        up1 = self.up1(up2)          # 128
        up1 = self.ref1(up1 + enc1)  # add skip and refine
        
        up0 = self.up0(up1)          # 256
        up0 = self.ref0(up0)         # no skip (since enc0 doesn't exist)
        
        out = torch.sigmoid(self.final(up0))
        return out

if __name__ == '__main__':
    model = UNetSR()
    print(model)
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    print('Input shape:', x.shape)
    print('Output shape:', y.shape)
