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
        # 'growth_rate': number of new feature maps produced by each DenseLayer
        # 'block_config': tuple defining how many DenseLayers are present in each DenseBlock (4 blocks total)

        super(DenseNet, self).__init__()

        # --- Initial Convolutional Layer ---
        # 1. input has 1 channel (grayscale X-ray)
        # 2. output is 64 feature maps
        # 3. using a large 7x7 kernel for broad receptive field in early stage
        # 4. stride=2 and padding=3 to reduce spatial size while preserving image center
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 5. batch normalization applied after initial conv for feature stability
        self.bn1 = nn.BatchNorm2d(64)

        # 6. max pooling with 3x3 kernel and stride=2 for further spatial downsampling
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Channel Tracker ---
        # tracks current number of feature maps flowing through the network
        num_channels = 64  # set after initial convolution

        # --- Containers for DenseBlocks and TransitionLayers ---
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        # --- Constructing Dense Blocks + Transitions ---
        for i, num_layers in enumerate(block_config):
            # 1. build a DenseBlock using the current channel count and growth rate
            block = DenseBlock(num_layers, num_channels, growth_rate)
            self.blocks.append(block)

            # 2. after adding the DenseBlock, update number of channels:
            #    each layer adds 'growth_rate' channels
            num_channels += num_layers * growth_rate

            # 3. insert a TransitionLayer after each block except the final one
            if i != len(block_config) - 1:
                # compress channels to half using transition
                trans = TransitionLayer(num_channels, num_channels // 2)
                self.transitions.append(trans)
                num_channels = num_channels // 2

        # --- Final Normalization ---
        # batch normalization before classification heads to clean up final features
        self.bn_final = nn.BatchNorm2d(num_channels)

        # --- Dual Classification Heads ---

        # Fracture Detection Head
        # this head outputs a single probability for binary classification
        self.fracture_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
            nn.Flatten(),                  # flatten to 1D vector
            nn.Linear(num_channels, 2),   # linear layer → 2 output
            nn.Sigmoid()                  # sigmoid for probability (0 to 1)
        )

        # Chest Condition Classification Head
        # this head outputs logits for 15 mutually exclusive chest conditions
        self.chest_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
            nn.Flatten(),                  # flatten to 1D vector
            nn.Linear(num_channels, 15)   # 15 classes → no softmax (handled in loss)
        )

    def forward(self, x):
        # --- Initial Feature Extraction ---
        # 1. apply initial convolution, batch normalization, ReLU activation, and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # --- Dense Blocks with Transitions ---
        for i, block in enumerate(self.blocks):
            # 2. pass input through current DenseBlock
            x = block(x)

            # 3. if not the last block, pass through corresponding TransitionLayer
            if i < len(self.transitions):
                x = self.transitions[i](x)

        # --- Final Processing ---
        # 4. apply final batch normalization and activation
        x = F.relu(self.bn_final(x))

        # --- Dual Output Heads ---
        # 5. fracture head produces a binary probability
        fracture_pred = self.fracture_head(x)

        # 6. chest head produces logits for 15 classes
        chest_pred = self.chest_head(x)

        # 7. return both predictions
        return fracture_pred, chest_pred
