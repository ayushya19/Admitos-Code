"""
model.py
========
Three-stream CNN with late fusion for radar human activity classification.

Architecture per stream:
    Input: [B, 1, 128, 128]  (each representation is single-channel)
    Block 1: Conv 3x3 (32)  → BN → ReLU → Conv 3x3 (32)  → BN → ReLU → MaxPool 2x2
    Block 2: Conv 3x3 (64)  → BN → ReLU → Conv 3x3 (64)  → BN → ReLU → MaxPool 2x2
    Block 3: Conv 3x3 (128) → BN → ReLU → Conv 3x3 (128) → BN → ReLU → MaxPool 2x2
    Block 4: Conv 3x3 (256) → BN → ReLU → Conv 3x3 (256) → BN → ReLU → AdaptiveAvgPool
    FC head: Linear(256 → STREAM_FEATURE_DIM) → BN → ReLU → Dropout

Late fusion:
    Concatenate 3 stream feature vectors → [B, 3 * STREAM_FEATURE_DIM]
    Classifier: Linear → BN → ReLU → Dropout → Linear → NUM_CLASSES

Streams:
    Stream 0 → Spectrogram   (channel 0)
    Stream 1 → Range-Time    (channel 1)
    Stream 2 → Range-Doppler (channel 2)
"""

import torch
import torch.nn as nn
import config


# ══════════════════════════════════════════════════════════════════════
# 1.  CONVOLUTIONAL BLOCK  (Conv → BN → ReLU → Conv → BN → ReLU)
# ══════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Double conv block: (Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════
# 2.  SINGLE STREAM CNN
# ══════════════════════════════════════════════════════════════════════
class StreamCNN(nn.Module):
    """
    Processes one radar representation (Spectrogram / Range-Time / Range-Doppler).
    Input  : [B, 1, H, W]
    Output : [B, STREAM_FEATURE_DIM]  (feature vector)

    Spatial flow for 128x128 input:
        After Block1 + MaxPool : 64x64
        After Block2 + MaxPool : 32x32
        After Block3 + MaxPool : 16x16
        After Block4 + AvgPool :  1x1
        Flatten                : 256-d
        FC                     : STREAM_FEATURE_DIM-d
    """

    def __init__(self, feature_dim: int = config.STREAM_FEATURE_DIM,
                 dropout: float = config.DROPOUT):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1:  1 → 32
            ConvBlock(1, 32),
            nn.MaxPool2d(2, 2),            # 128→64

            # Block 2: 32 → 64
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),            # 64→32

            # Block 3: 64 → 128
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),            # 32→16

            # Block 4: 128 → 256
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),  # 16→1
        )

        # FC head: project to stream feature dimension
        self.fc_head = nn.Sequential(nn.Flatten(), nn.Linear(256, feature_dim, bias=False), # [B, 256]
                                    nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True),nn.Dropout(dropout))

    def forward(self, x):
        # x: [B, 1, H, W]
        x = self.encoder(x)    # [B, 256, 1, 1]
        x = self.fc_head(x)    # [B, feature_dim]
        return x


# ══════════════════════════════════════════════════════════════════════
# 3.  THREE-STREAM MODEL WITH LATE FUSION
# ══════════════════════════════════════════════════════════════════════
class ThreeStreamCNN(nn.Module):
    """
    Three independent StreamCNN encoders (one per radar representation),
    followed by late fusion (concatenation) and a shared classifier head.

    Input  : x  [B, 3, H, W]
                 x[:,0,:,:]  → Spectrogram
                 x[:,1,:,:]  → Range-Time
                 x[:,2,:,:]  → Range-Doppler

    Output : logits  [B, NUM_CLASSES]
    """

    def __init__(self,num_classes: int = config.NUM_CLASSES, feature_dim: int = config.STREAM_FEATURE_DIM, dropout: float = config.DROPOUT):
        super().__init__()

        # Three independent streams
        self.stream_spec = StreamCNN(feature_dim, dropout) # Spectrogram
        self.stream_rt = StreamCNN(feature_dim, dropout) # Range-Time
        self.stream_rd = StreamCNN(feature_dim, dropout) # Range-Doppler

        # Late fusion classifier
        # Fused dim = 3 * feature_dim
        fused_dim = 3 * feature_dim

        self.classifier = nn.Sequential(nn.Linear(fused_dim, fused_dim//2, bias=False), nn.BatchNorm1d(fused_dim//2),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(fused_dim//2, fused_dim//4, bias=False), 
            nn.BatchNorm1d(fused_dim//4), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(fused_dim//4, num_classes))

    def forward(self, x):
        # Split into 3 single-channel inputs
        x_spec = x[:, 0:1, :, :]   # [B, 1, H, W]
        x_rt   = x[:, 1:2, :, :]   # [B, 1, H, W]
        x_rd   = x[:, 2:3, :, :]   # [B, 1, H, W]

        # Pass through each stream independently
        f_spec = self.stream_spec(x_spec)   # [B, feature_dim]
        f_rt   = self.stream_rt(x_rt)       # [B, feature_dim]
        f_rd   = self.stream_rd(x_rd)       # [B, feature_dim]

        # Late fusion: concatenate feature vectors
        fused = torch.cat([f_spec, f_rt, f_rd], dim=1)   # [B, 3*feature_dim]

        # Classify
        logits = self.classifier(fused)   # [B, num_classes]

        return logits

# ══════════════════════════════════════════════════════════════════════
# 4.  MODEL SUMMARY UTILITY
# ══════════════════════════════════════════════════════════════════════
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ══════════════════════════════════════════════════════════════════════
# 5.  SANITY CHECK
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = ThreeStreamCNN().to(device)

    # Dummy batch: [B=4, 3 channels, 128x128]
    dummy = torch.randn(4, 3, 128, 128).to(device)
    logits = model(dummy)

    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {logits.shape}")          # expect [4, 6]
    print(f"  Trainable parameters: {count_parameters(model):,}")
    print("model.py sanity check passed.")
