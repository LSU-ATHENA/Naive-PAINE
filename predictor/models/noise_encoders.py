import torch
import torch.nn as nn


class CustomNoiseEncoder(nn.Module):

    def __init__(self, in_channels: int = 4, spatial_size: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.SiLU()
        self.do1 = nn.Dropout2d(0.3)
        self.skip_1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)

        self.ds1 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.ds_bn1 = nn.BatchNorm2d(64)
        self.ds_act1 = nn.SiLU()
        self.ds_do1 = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU()
        self.do2 = nn.Dropout2d(0.3)

        self.ds2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.ds_bn2 = nn.BatchNorm2d(128)
        self.ds_act2 = nn.SiLU()
        self.ds_do2 = nn.Dropout2d(0.3)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.SiLU()
        self.do3 = nn.Dropout2d(0.3)

        self.ds3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.ds_bn3 = nn.BatchNorm2d(256)
        self.ds_act3 = nn.SiLU()
        self.ds_do3 = nn.Dropout2d(0.3)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.SiLU()
        self.do4 = nn.Dropout2d(0.3)

        self.ds4 = nn.Conv2d(256, 1024, kernel_size=5, stride=2, padding=2)
        self.ds_bn4 = nn.BatchNorm2d(1024)
        self.ds_act4 = nn.SiLU()
        self.ds_do4 = nn.Dropout2d(0.3)

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.final_do = nn.Dropout(0.3)

        self._output_dim = 1024

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_1(x)
        x = self.do1(self.act1(self.bn1(self.conv1(x)))) + identity
        x = self.ds_do1(self.ds_act1(self.ds_bn1(self.ds1(x))))
        x = self.do2(self.act2(self.bn2(self.conv2(x)))) + x
        x = self.ds_do2(self.ds_act2(self.ds_bn2(self.ds2(x))))
        x = self.do3(self.act3(self.bn3(self.conv3(x)))) + x
        x = self.ds_do3(self.ds_act3(self.ds_bn3(self.ds3(x))))
        x = self.do4(self.act4(self.bn4(self.conv4(x)))) + x
        x = self.ds_do4(self.ds_act4(self.ds_bn4(self.ds4(x))))

        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.final_do(x)
        return x


def get_noise_encoder(spatial_size: int = 128, **kwargs) -> nn.Module:
    return CustomNoiseEncoder(spatial_size=spatial_size, **kwargs)
