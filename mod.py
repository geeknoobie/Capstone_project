import torch
import torch.nn as nn
import torch.nn.functional as F
# 1. First we have our DenseLayer
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        # 'in_channels': number of input features.
        # 'growth_rate': number of new features this layer will produce.
        super(DenseLayer, self).__init__()

        # First transformation (Bottleneck layer).
        # Normalizes the input features.
        self.bn1 = nn.BatchNorm2d(in_channels)
        # Applies a 1x1 convolution to reduce the number of feature maps (bottleneck), expanding to 4 times the growth rate.
        self.conv1 = nn.Conv2d(in_channels,
                               4 * growth_rate,
                               kernel_size=1,
                               bias=False)  # No bias needed with BatchNorm.

        # Second transformation.
        # Normalizes the output of the bottleneck layer.
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        # Applies a 3x3 convolution to produce 'growth_rate' new feature maps.
        # Padding is used to maintain the spatial dimensions of the input.
        self.conv2 = nn.Conv2d(4 * growth_rate,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)  # No bias needed with BatchNorm.

    def forward(self, x):
        # Apply transformations and concatenate with input.
        # 1. Apply batch normalization to the input, then ReLU activation, and then the 1x1 convolution.
        out = self.conv1(F.relu(self.bn1(x)))
        # 2. Apply batch normalization to the output of the 1x1 convolution, then ReLU activation, and then the 3x3 convolution.
        out = self.conv2(F.relu(self.bn2(out)))
        # 3. Concatenate the original input 'x' with the output 'out' along the channel dimension (dimension 1).
        out = torch.cat([x, out], 1)
        # 4. Return the concatenated output.
        return out
# 2. Then our DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        # num_layers: how many dense layers in this block (e.g., 6, 12, 24, or 16).
        # in_channels: number of input channels to the block.
        # growth_rate: how many new features each layer creates.
        super(DenseBlock, self).__init__()

        # Create a list to hold all layers.
        self.layers = nn.ModuleList()

        # Create each layer.
        for i in range(num_layers):
            # For each new layer:
            # Input channels = initial channels + new features from previous layers.
            # new features from previous layers = i * growth_rate.
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def forward(self, x):
        # Pass input through each layer sequentially.
        # Each layer's output becomes input for next layer.
        # Due to concatenation in DenseLayer, features accumulate.
        for layer in self.layers:
            x = layer(x)
        return x
# 3. Next, we add the Transition Layer
class TransitionLayer(nn.Module):
    """
    Transition Layer between Dense Blocks:
    - Reduces spatial dimensions.
    - Reduces number of channels.
    """

    def __init__(self, in_channels, out_channels):
        # in_channels: number of input features from previous DenseBlock.
        # out_channels: number of output features (typically in_channels // 2).
        super(TransitionLayer, self).__init__()

        # Components of transition layer.
        # Normalizes the input features.
        self.bn = nn.BatchNorm2d(in_channels)
        # Reduces the number of channels using a 1x1 convolution.
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              bias=False)  # No bias with BatchNorm.
        # Reduces the spatial dimensions (height and width) by half using average pooling.
        self.pool = nn.AvgPool2d(kernel_size=2,
                                 stride=2)

    def forward(self, x):
        # Apply transformations sequentially.
        # 1. Apply batch normalization to the input, then ReLU activation, and then the 1x1 convolution (channel reduction).
        x = self.conv(F.relu(self.bn(x)))
        # 2. Apply average pooling to reduce the spatial dimensions.
        x = self.pool(x)
        # 3. Return the transformed output.
        return x
# 4. Finally, we create the complete DenseNet
class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16)):
        # growth_rate: number of new features each layer produces.
        # block_config: tuple defining number of layers in each DenseBlock.
        super(DenseNet, self).__init__()

        # Initial Convolution Layer.
        # Input: 1 channel (grayscale X-ray).
        # Output: 64 feature maps.
        # Large kernel for initial features.
        # Reduce spatial dimensions.
        # Maintain spatial dimensions.
        # No bias needed with BatchNorm.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Normalize initial features.
        self.bn1 = nn.BatchNorm2d(64)
        # Further reduce spatial dimensions.
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Track number of channels.
        num_channels = 64  # Starting number after initial conv.

        # Create lists to hold all blocks and transitions.
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        # Create DenseBlocks and TransitionLayers.
        for i, num_layers in enumerate(block_config):
            # Add a DenseBlock.
            # Number of layers in this block.
            # Current number of channels.
            # How many new features per layer.
            block = DenseBlock(num_layers, num_channels, growth_rate)
            self.blocks.append(block)

            # Update channel count.
            num_channels += num_layers * growth_rate

            # Add a transition layer after each block (except the last).
            # Current channels.
            # Reduce channels by half.
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_channels, num_channels // 2)
                self.transitions.append(trans)
                num_channels = num_channels // 2

        # Final BatchNorm.
        self.bn_final = nn.BatchNorm2d(num_channels)

        # Fracture detection head.
        self.fracture_head = nn.Sequential(
            # Global average pooling.
            nn.AdaptiveAvgPool2d((1, 1)),
            # Flatten for linear layer.
            nn.Flatten(),
            # Single output for binary classification.
            nn.Linear(num_channels, 1),
            # Sigmoid for 0-1 probability.
            nn.Sigmoid()
        )

        # Chest conditions head.
        self.chest_head = nn.Sequential(
            # Global average pooling.
            nn.AdaptiveAvgPool2d((1, 1)),
            # Flatten for linear layer.
            nn.Flatten(),
            # 15 outputs for chest conditions.
            nn.Linear(num_channels, 15),
        )

    def forward(self, x):
        # Initial processing.
        # Apply the initial convolution, batch normalization, ReLU, and max pooling.
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Pass through each DenseBlock and TransitionLayer.
        for i, block in enumerate(self.blocks):
            # Pass through DenseBlock.
            x = block(x)
            if i < len(self.transitions):
                # Pass through TransitionLayer.
                x = self.transitions[i](x)

        # Final processing.
        # Apply final batch normalization and ReLU.
        x = F.relu(self.bn_final(x))

        # Get predictions from both heads.
        fracture_pred = self.fracture_head(x)
        chest_pred = self.chest_head(x)

        return fracture_pred, chest_pred