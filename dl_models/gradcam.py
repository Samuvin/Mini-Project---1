"""
Grad-CAM implementation for 1-D SE-ResNet convolutional layers.

Computes class-discriminative importance scores for each input feature
by back-propagating the gradient of the predicted class into the last
convolutional layer and weighting activations by gradient magnitude.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization," ICCV 2017.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class GradCAM1D:
    """Grad-CAM for 1-D convolutional feature maps.

    Usage::

        cam = GradCAM1D(model, target_layer=model.speech_encoder.block2.conv2)
        importance = cam(speech, handwriting, gait)
        # importance is a dict with keys 'speech', 'handwriting', 'gait'
        # each value is a numpy array of shape [num_features] with
        # per-feature importance scores (non-negative, sum to 1).

    Args:
        model: The ``MultimodalPDNet`` instance (must be in eval mode).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._activations: dict[str, torch.Tensor] = {}
        self._gradients: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    # -- hook helpers ------------------------------------------------- #

    def _register_hooks(self) -> None:
        """Register forward/backward hooks on the last conv layer of
        each modality encoder."""
        self._remove_hooks()

        targets = {
            "speech": self.model.speech_encoder.block2.conv2,
            "handwriting": self.model.handwriting_encoder.block2.conv2,
            "gait": self.model.gait_encoder.block2.conv2,
        }

        for name, layer in targets.items():
            self._hooks.append(
                layer.register_forward_hook(self._make_fwd_hook(name))
            )
            self._hooks.append(
                layer.register_full_backward_hook(self._make_bwd_hook(name))
            )

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_fwd_hook(self, name: str):
        def hook(_module, _input, output):
            self._activations[name] = output.detach()
        return hook

    def _make_bwd_hook(self, name: str):
        def hook(_module, _grad_input, grad_output):
            self._gradients[name] = grad_output[0].detach()
        return hook

    # -- public API --------------------------------------------------- #

    @torch.enable_grad()
    def __call__(
        self,
        speech: torch.Tensor,
        handwriting: torch.Tensor,
        gait: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute per-feature Grad-CAM importance for each modality.

        All inputs should be single-sample tensors [1, num_features].
        The model is temporarily set to train mode for gradient
        computation, then restored.

        Args:
            speech: [1, 22] speech features.
            handwriting: [1, 10] handwriting features.
            gait: [1, 10] gait features.
            target_class: If ``None`` (default) uses the predicted class.

        Returns:
            Dict mapping modality name to a 1-D numpy array of
            non-negative importance scores (sums to 1).
        """
        self._register_hooks()
        was_training = self.model.training
        self.model.eval()  # BN/Dropout in eval, but we need grads

        # Enable grads on inputs
        speech = speech.clone().requires_grad_(True)
        handwriting = handwriting.clone().requires_grad_(True)
        gait = gait.clone().requires_grad_(True)

        # Forward
        out = self.model(speech, handwriting, gait)
        logit = out["logit"].squeeze()

        # Target class
        if target_class is None:
            target_class = 1 if logit.item() > 0 else 0

        # Backward w.r.t. target class
        self.model.zero_grad()
        if target_class == 1:
            logit.backward()
        else:
            (-logit).backward()

        # Compute Grad-CAM for each modality
        result: dict[str, np.ndarray] = {}
        modality_features = {
            "speech": speech.shape[-1],
            "handwriting": handwriting.shape[-1],
            "gait": gait.shape[-1],
        }

        for name, num_feats in modality_features.items():
            act = self._activations.get(name)
            grad = self._gradients.get(name)
            if act is None or grad is None:
                result[name] = np.ones(num_feats) / num_feats
                continue

            # act, grad: [1, C, L]
            # Channel-wise importance weights (global average of gradients)
            weights = grad.mean(dim=-1, keepdim=True)  # [1, C, 1]
            # Weighted combination of activations
            cam = (weights * act).sum(dim=1).squeeze(0)  # [L]
            # ReLU -- keep only positive contributions
            cam = torch.relu(cam)

            cam_np = cam.cpu().numpy()
            # Interpolate/resize to match original feature count
            if len(cam_np) != num_feats:
                cam_np = np.interp(
                    np.linspace(0, 1, num_feats),
                    np.linspace(0, 1, len(cam_np)),
                    cam_np,
                )

            # Normalize to [0, 1] range
            total = cam_np.sum()
            if total > 0:
                cam_np = cam_np / total
            else:
                cam_np = np.ones(num_feats) / num_feats

            result[name] = cam_np

        # Cleanup
        self._remove_hooks()
        if was_training:
            self.model.train()

        return result
