"""
Adding a new model needs to modify:
- CONFIG in config.py
- get_model() in model.py
- compute_loss_train() in train.py
- compute_score() in evaluate.py


"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from util import count_model_parameters, my_print, get_saved_info, range01, get_final_model_id
from config import CONFIG, MODEL_PATH, DEVICE, N_CHN_PAIR, N_FREQ

# <editor-fold desc="======== Visual Transformer model =======">
# ==================== Visual Transformer model ====================
class CoSP_ViT_Model(nn.Module):
    def __init__(self,
                 image_size=128,
                 patch_size=8,
                 in_channels=1,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1,
                 dropout_rate=0.1):
        super().__init__()

        # Base ViT model (patch16, but we override it)
        import timm  # Optional dependency for the experimental ViT model.

        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            img_size=image_size,
            in_chans=in_channels
        )

        # Override patch embedding for different patch size
        self.vit.patch_embed.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Recalculate number of patches
        n_patches = (image_size // patch_size) ** 2

        # Reinitialize position embedding (CLS token + patches)
        self.vit.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.vit.pos_embed, std=0.02)

        # Update classifier head
        self.vit.head = nn.Linear(embed_dim, 1)
        self.image_size = image_size

    def forward(self, x):
        """
        Input shape: (B, 1, H, W) e.g., (B, 1, 128, 128)
        Output shape: (B,) — logit for binary classification
        """
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.pad(x, (0, self.image_size - x.shape[-1],
                          0, self.image_size - x.shape[-2]), mode='constant', value=0)

        return self.vit(x)
# </editor-fold>



# <editor-fold desc="====== CoSP model ========">
# ==================== CoSP model ====================
class CoSP(nn.Module):
    def __init__(self, hidden_sizes, kernel_sizes, n_features, n_frequencies):
        super(CoSP, self).__init__()

        self.layers = nn.ModuleList()
        in_channels = 1
        for out_channels, kernel_size in zip(hidden_sizes, kernel_sizes):
            padding = kernel_size // 2
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        # Dummy forward pass to determine flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_features, n_frequencies)
            for layer in self.layers:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                    dummy_input = layer(dummy_input)
            flattened_size = dummy_input.view(dummy_input.size(0), -1).shape[1]

        # Define fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        # x: (batch_size, n_frequencies, n_channels), (32,109,120)
        # Reshape x to (batch_size, 1, n_frequencies, n_features) (32,1,109,120)
        # x = x.unsqueeze(1)

        # Pass input through each convolutional layer
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
# </editor-fold>


# <editor-fold desc="====== MBG-CoSP model ========">
# ==================== MBG-CoSP (Multi-Branch Gated CoSP) Model ====================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AsymCNNBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, orientation):
        super().__init__()
        if orientation == 'horizontal':  # 1 × n
            k = (1, kernel_size)
        elif orientation == 'vertical':  # n × 1
            k = (kernel_size, 1)
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.se(x)
        return x


class SquareCNNBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.se(x)
        return x


class GatedFusion(nn.Module):
    def __init__(self, n_branches, in_channels, height, width):
        super().__init__()
        self.gates = nn.Sequential(
            nn.Linear(in_channels * n_branches, n_branches),
            nn.Softmax(dim=1)
        )

    def forward(self, feature_list):
        # Each feature in feature_list should be [B, C, H, W]
        B, C, H, W = feature_list[0].shape
        N = len(feature_list)

        # Stack: [B, N, C, H, W]
        stacked = torch.stack(feature_list, dim=1)  # [B, N, C, H, W]
        pooled = F.adaptive_avg_pool2d(stacked.view(B * N, C, H, W), 1)  # [B*N, C, 1, 1]
        pooled = pooled.view(B, N, C)  # [B, N, C]
        flat = pooled.view(B, -1)  # [B, N*C]

        weights = self.gates(flat)  # [B, N]
        weights = torch.softmax(weights, dim=1)  # [B, N]

        # Fuse branches using soft weights
        fused = 0
        for i in range(N):
            w = weights[:, i].view(B, 1, 1, 1)  # [B, 1, 1, 1]
            fused += w * feature_list[i]
        return fused


class MBGCoSP(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=5, height=109, width=120):
        super().__init__()
        self.branch_h = AsymCNNBranch(in_channels, out_channels, kernel_size, 'horizontal')
        self.branch_v = AsymCNNBranch(in_channels, out_channels, kernel_size, 'vertical')
        self.branch_s = SquareCNNBranch(in_channels, out_channels, kernel_size)

        self.fusion = GatedFusion(n_branches=3, in_channels=out_channels, height=height, width=width)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, 1)  # Binary classification (logit)
        )

    def forward(self, x):
        out_h = self.branch_h(x)
        out_v = self.branch_v(x)
        out_s = self.branch_s(x)
        fused = self.fusion([out_h, out_v, out_s])
        pooled = self.global_pool(fused)
        return self.classifier(pooled)


# </editor-fold>


# ==================== FCN model ====================
# Define the fully connected network (FCN)
class FCN(nn.Module):
    def __init__(self, n_freq, n_chn_pair, hidden_sizes):
        super(FCN, self).__init__()
        input_size = n_freq * n_chn_pair
        self.flatten = nn.Flatten()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))

            # Add BatchNorm for first three layers (optional logic)
            if i < 3:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.ReLU())

            # Add dropout after second and later layers
            if i >= 1:
                layers.append(nn.Dropout(0.5))

            prev_size = hidden_size

        # Final output layer: 1 logit
        layers.append(nn.Linear(prev_size, 1))

        # Combine all into a Sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


#==================== utility ====================
def print_model_architecture(model, input_shape=(1, 1, 128, 128), save_path=None):
    """
    Print and optionally save the architecture summary of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to inspect.
        input_shape (tuple): The shape of the input tensor.
        save_path (str or Path, optional): If provided, save the summary to this file.
    """
    from torchinfo import summary

    info = summary(model, input_size=input_shape, verbose=1, col_names=["input_size", "output_size", "num_params"])

    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(str(info))
        print(f"\n✅ Model summary saved to: {save_path}")
    # else:
    #     print(info)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # init.xavier_normal_(m.weight)
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            # Initialize weights and biases for LSTM layers
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Initialize the forget gate bias to 1 (optional, can help with training)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)


def get_model(model_type='CoSP', model_id=None, config=None, device=DEVICE):
    if isinstance(device, int):
        # Convert the rank (integer) to a proper device string
        device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'

    if model_type == 'CoSP':
        if model_id is not None:
            config = get_saved_info(model_id, 'cfg')
        model = CoSP(hidden_sizes=config['hidden_size'],
                                    kernel_sizes=config['kernel_size'],
                                    n_features=N_CHN_PAIR,
                                    n_frequencies=N_FREQ)
        if model_id is not None:
            f_model = MODEL_PATH / f"model{model_id['model_id']}_round{model_id['round_id']}.pth"
            model.load_state_dict(torch.load(f_model, map_location=device))
        else:
            initialize_weights(model)

    elif model_type == 'FCN':
        model = FCN(n_freq=N_FREQ, n_chn_pair=N_CHN_PAIR, hidden_sizes=config['hidden_size'])
        if model_id is not None:
            f_model = MODEL_PATH / f"model{model_id['model_id']}_round{model_id['round_id']}.pth"
            model.load_state_dict(torch.load(f_model, map_location=device))

    elif model_type == 'ViT':
        model = CoSP_ViT_Model(
            image_size=config['vit_image_size'],
            patch_size=config['vit_patch_size'],
            in_channels=config['vit_in_channels'],
            num_classes=1
        )
        if model_id is not None:
            f_model = MODEL_PATH / f"model{model_id['model_id']}_round{model_id['round_id']}.pth"
            model.load_state_dict(torch.load(f_model, map_location=device))


    elif model_type == 'MBGCoSP':
        model = MBGCoSP(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            height=N_FREQ,
            width=N_CHN_PAIR

        )
    else:
        raise ValueError(f"ERROR: unknown model_type={model_type}")


    model.to(device)

    first_param_device = next(model.parameters()).device
    print(f"Model is on device: {first_param_device}")

    # if device is not None: model.to(device)
    return model


def model_check():
    input_size = CONFIG['n_chn']
    x = torch.randn(64, 4000, input_size)  # Batch x Seq x Feature
    model = get_model(device='cpu')
    print(model)
    n_para_all, n_para_train = count_model_parameters(model)
    my_print(1, f"\n model parameters: n_para_all={n_para_all}, n_para_trainable={n_para_train}")
    out = model(x)
    my_print(1, f"\n output_shape={out.shape}")



def plot_conv_weights(model, layer_idx, save=False, model_id=None):
    '''
    Function to plot the learned parameters of the convolutional layers
    layer_num=0: Conv Layer 1: 32 filters, kernel size 7
    layer_num=1: Conv Layer 2: 64 filters, kernel size 5
    layer_num=2: Conv Layer 3: 128 filters, kernel size 3
    layer_num=3: Conv Layer 4: 256 filters, kernel size 3

    Darker color: Higher weight values (more importance or emphasis on features).
    Lighter color: Lower weight values (less importance or suppression of features).
    '''

    conv_layer = None
    i = 0
    # Loop over the layers and select the specified convolutional layer
    for layer in model.layers:
        if isinstance(layer, nn.Conv2d):
            if i == layer_idx:
                conv_layer = layer
                break
            i += 1

    if conv_layer is None:
        print(f"Conv layer {layer_idx} not found")
        return

    # Get the weights of the convolutional layer
    weights = conv_layer.weight.data.cpu().numpy()

    # Plot the weights for each filter
    num_filters = weights.shape[0]
    num_cols = math.ceil(math.sqrt(num_filters))
    num_rows = math.ceil(num_filters / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    axes = axes.flatten()  # Flatten axes to easily iterate through them
    for i, ax in enumerate(axes):
        if i < num_filters:
            normalized_weight = 1 - range01(weights[i, 0, :, :])
            ax.imshow(normalized_weight, cmap='gray')
            if i == 0:
                filter_shape = weights[i, 0, :, :].shape
        ax.axis('off')

    plt.suptitle(f"CONV{layer_idx+1}: {num_filters} Filters, {filter_shape}", fontsize=16)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save:
        save_path = MODEL_PATH / f"model{model_id}_conv{layer_idx+1}_filters.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()



def plot_fc_weights(model, layer_idx, save=False, model_id=None):
    # visualize the fully connected layer's parameters
    if layer_idx == 0:
        fc_weights = model.fc1.weight.data.cpu().numpy()
        plt.figure(figsize=(10, 15))
    elif layer_idx == 1:
        fc_weights = model.fc2.weight.data.cpu().numpy()
        plt.figure(figsize=(10, 2))
    w_shape = fc_weights.shape

    plt.imshow(fc_weights, aspect='auto', cmap='viridis')
    plt.colorbar()
    if w_shape[0] == 1:
        ax = plt.gca()
        ax.set_yticks([])
    plt.title(f"FC{layer_idx+1}: {w_shape}")

    # Save the plot if save_path is provided
    if save:
        save_path = MODEL_PATH / f"model{model_id}_fc{layer_idx + 1}_weights.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
