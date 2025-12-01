import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm(self.cnn(x)))


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_cnn,
        pool=False,
        upsample=False,
        softmax=False,
    ):
        super(Block, self).__init__()
        self.softmax = softmax
        self.pool = pool
        self.upsample = upsample
        if num_cnn == 2:
            self.block = nn.ModuleList(
                [
                    CNNBlock(in_channels=in_channels, out_channels=out_channels),
                    CNNBlock(in_channels=out_channels, out_channels=out_channels),
                ]
            )
        if num_cnn == 3:
            self.block = nn.ModuleList(
                [
                    CNNBlock(in_channels=in_channels, out_channels=out_channels),
                    CNNBlock(in_channels=out_channels, out_channels=out_channels),
                    CNNBlock(in_channels=out_channels, out_channels=out_channels),
                ]
            )

        self.mp = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.mup = nn.MaxUnpool2d(2, stride=2)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x, ind=None):

        if self.upsample:
            x = self.mup(x, ind)
            # print(f"UPSAMPLE {x.shape}")

        for module in self.block:
            x = module(x)

        if self.pool:
            x, ind = self.mp(x)
            # print(f"POOL {x.shape}")
            return x, ind

        if self.softmax:
            x = self.sm(x)
        # print(f"ELSE {x.shape}")
        return x


class SegNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=32, num_two=2, num_blocks=5, channel_step=64
    ):
        super(SegNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.poolblock = nn.ModuleList(
            [
                (
                    Block(
                        in_channels=(
                            channel_step * 2 ** (i - 1) if i != 0 else in_channels
                        ),
                        out_channels=channel_step * 2**i,
                        num_cnn=2 if i < num_two else 3,
                        pool=True,
                    )
                )
                for i in range(num_blocks)
            ]
        )
        self.unpoolblock = nn.ModuleList(
            [
                Block(
                    in_channels=(channel_step * 2 ** (4 - i)),
                    out_channels=(
                        channel_step * 2 ** (3 - i)
                        if i != num_blocks - 1
                        else out_channels
                    ),
                    num_cnn=2 if i < num_two else 3,
                    upsample=True,
                    softmax=True if i == num_blocks - 1 else False,
                    # softmax=False,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x):
        index_list = []
        # print(self.unpoolblock)
        for module in self.poolblock:
            x, ind = module(x)
            index_list.append(ind)
        for i, module in enumerate(self.unpoolblock):
            ind = index_list[4 - i]
            # print(x.shape, ind.shape)
            x = module(x, ind)
        return x
