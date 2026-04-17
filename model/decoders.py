import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNetStyleDecoder(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        C1, C2, C3, C4 = channels
        self.bottleneck = nn.Conv2d(C4, 256, kernel_size=1)
        self.up3 = UpBlock(in_channels=256, skip_channels=C3, out_channels=256)
        self.up2 = UpBlock(in_channels=256, skip_channels=C2, out_channels=128)
        self.up1 = UpBlock(in_channels=128, skip_channels=C1, out_channels=64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, hidden_states):
        s1, s2, s3, s4 = hidden_states
        x = self.bottleneck(s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        logits_1_4 = self.classifier(x)
        return logits_1_4

class UpsampleRefinement(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, 3, padding=1),
            nn.BatchNorm2d(num_classes)
        )

    def forward(self, x):
        return self.refine(x)
