"""
SE-ResNet 1D CNN with Attention-Based Multimodal Fusion.

Architecture:
    Per-modality SE-ResNet1D encoders (speech 22-d, handwriting 10-d,
    gait 10-d) produce 64-d embeddings.  An attention fusion layer
    learns modality importance weights and produces a single 64-d
    fused vector.  A dense classifier maps to a binary prediction
    with a confidence score.

References:
    He et al., "Deep Residual Learning," CVPR 2016.
    Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Squeeze-and-Excitation Block (1-D)                                 #
# ------------------------------------------------------------------ #

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D feature maps.

    Squeezes global spatial information into a channel descriptor via
    adaptive average pooling, then excites (re-weights) channels through
    a two-layer bottleneck with sigmoid gating.

    Args:
        channels: Number of input channels.
        reduction: Bottleneck reduction ratio (default 4).
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (re-weighted x, channel weights [B, C])."""
        b, c, _ = x.size()
        # Squeeze: [B, C, L] -> [B, C, 1] -> [B, C]
        se = self.squeeze(x).view(b, c)
        # Excitation: [B, C] -> [B, C]
        weights = self.excitation(se)
        # Scale: [B, C, 1] * [B, C, L]
        out = x * weights.unsqueeze(-1)
        return out, weights


# ------------------------------------------------------------------ #
#  Residual SE Block (1-D)                                            #
# ------------------------------------------------------------------ #

class ResidualSEBlock1D(nn.Module):
    """ResNet residual block with SE channel attention for 1-D data.

    Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> SE -> + skip -> ReLU

    A 1x1 convolution is used on the skip path when in_channels
    differs from out_channels.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        reduction: SE bottleneck ratio (default 4).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1D(out_channels, reduction)

        # Skip (shortcut) connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (block output, SE channel weights)."""
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # SE recalibration
        out, se_weights = self.se(out)

        # Residual addition
        out = out + identity
        out = F.relu(out, inplace=True)
        return out, se_weights


# ------------------------------------------------------------------ #
#  Per-Modality SE-ResNet1D Encoder                                   #
# ------------------------------------------------------------------ #

class ModalitySEResNet1D(nn.Module):
    """SE-ResNet1D encoder for a single data modality.

    Architecture::

        Conv1d(1->32) + BN + ReLU
        ResidualSEBlock1D(32->32)
        ResidualSEBlock1D(32->64)
        AdaptiveAvgPool1d(1)
        Dropout -> Linear(64->embed_dim)

    Args:
        num_features: Length of the input feature vector (e.g. 22 for speech).
        embed_dim: Embedding dimensionality (default 64).
        reduction: SE reduction ratio (default 4).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        reduction: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_features = num_features

        # Initial convolution: lift 1 channel to 32
        self.initial = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Residual SE blocks
        self.block1 = ResidualSEBlock1D(32, 32, reduction, dropout)
        self.block2 = ResidualSEBlock1D(32, 64, reduction, dropout)

        # Global Average Pooling + projection
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, embed_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Feature vector [B, num_features] or [B, 1, num_features].

        Returns:
            embedding: [B, embed_dim] embedding vector.
            info: Dict with ``se_weights_1``, ``se_weights_2``,
                  and ``last_conv_output`` (for Grad-CAM).
        """
        # Ensure shape [B, 1, num_features]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.initial(x)

        out, se_w1 = self.block1(out)
        out, se_w2 = self.block2(out)

        last_conv = out  # keep reference for Grad-CAM

        out = self.gap(out).squeeze(-1)  # [B, 64]
        out = self.dropout(out)
        embedding = self.fc(out)  # [B, embed_dim]

        info = {
            "se_weights_1": se_w1,
            "se_weights_2": se_w2,
            "last_conv_output": last_conv,
        }
        return embedding, info


# ------------------------------------------------------------------ #
#  Attention-Based Modality Fusion                                    #
# ------------------------------------------------------------------ #

class AttentionFusion(nn.Module):
    """Modality-level attention fusion.

    Learns a scalar importance score for each modality embedding,
    applies softmax across modalities, and returns the weighted sum.

    Args:
        embed_dim: Dimensionality of each modality embedding.
        num_modalities: Number of modalities (default 3).
    """

    def __init__(self, embed_dim: int = 64, num_modalities: int = 3) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.num_modalities = num_modalities

    def forward(
        self, embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse modality embeddings.

        Args:
            embeddings: [B, num_modalities, embed_dim].

        Returns:
            fused: [B, embed_dim] weighted combination.
            weights: [B, num_modalities] attention weights (sum to 1).
        """
        # Compute per-modality scores: [B, num_modalities, 1]
        scores = self.attention(embeddings)
        # Softmax across modality dimension
        weights = F.softmax(scores, dim=1)  # [B, num_modalities, 1]
        # Weighted sum
        fused = (weights * embeddings).sum(dim=1)  # [B, embed_dim]
        return fused, weights.squeeze(-1)


# ------------------------------------------------------------------ #
#  Dense Classifier Head                                              #
# ------------------------------------------------------------------ #

class DenseClassifier(nn.Module):
    """Two-layer dense head with dropout.

    Args:
        embed_dim: Input dimensionality (default 64).
        hidden: Hidden layer size (default 32).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self, embed_dim: int = 64, hidden: int = 32, dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit [B, 1]."""
        return self.net(x)


# ------------------------------------------------------------------ #
#  Full Multimodal SE-ResNet Network                                  #
# ------------------------------------------------------------------ #

class MultimodalPDNet(nn.Module):
    """Multimodal Parkinson's Disease prediction network.

    Combines three SE-ResNet1D modality encoders, an attention fusion
    layer, and a dense classifier.

    Args:
        speech_features: Number of speech input features (default 22).
        handwriting_features: Number of handwriting features (default 10).
        gait_features: Number of gait features (default 10).
        embed_dim: Shared embedding dimensionality (default 64).
        reduction: SE reduction ratio (default 4).
        dropout: Global dropout probability (default 0.3).
    """

    def __init__(
        self,
        speech_features: int = 22,
        handwriting_features: int = 10,
        gait_features: int = 10,
        embed_dim: int = 64,
        reduction: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.speech_encoder = ModalitySEResNet1D(
            speech_features, embed_dim, reduction, dropout,
        )
        self.handwriting_encoder = ModalitySEResNet1D(
            handwriting_features, embed_dim, reduction, dropout,
        )
        self.gait_encoder = ModalitySEResNet1D(
            gait_features, embed_dim, reduction, dropout,
        )

        self.fusion = AttentionFusion(embed_dim, num_modalities=3)
        self.classifier = DenseClassifier(embed_dim, embed_dim // 2, dropout)

    def forward(
        self,
        speech: torch.Tensor,
        handwriting: torch.Tensor,
        gait: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            speech: [B, 22] speech feature vectors.
            handwriting: [B, 10] handwriting feature vectors.
            gait: [B, 10] gait feature vectors.

        Returns:
            Dictionary with keys:
                logit: [B, 1] raw logit.
                probability: [B, 1] sigmoid probability.
                attention_weights: [B, 3] modality attention weights.
                speech_info: SE weights and last conv from speech encoder.
                handwriting_info: Same for handwriting.
                gait_info: Same for gait.
        """
        s_emb, s_info = self.speech_encoder(speech)
        h_emb, h_info = self.handwriting_encoder(handwriting)
        g_emb, g_info = self.gait_encoder(gait)

        # Stack embeddings: [B, 3, embed_dim]
        stacked = torch.stack([s_emb, h_emb, g_emb], dim=1)

        fused, attn_weights = self.fusion(stacked)
        logit = self.classifier(fused)
        prob = torch.sigmoid(logit)

        return {
            "logit": logit,
            "probability": prob,
            "attention_weights": attn_weights,
            "speech_info": s_info,
            "handwriting_info": h_info,
            "gait_info": g_info,
        }
