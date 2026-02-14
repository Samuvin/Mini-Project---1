"""
PyTorch Dataset for multimodal Parkinson's Disease tabular data.

Loads speech (22 features), handwriting (10 features), and gait
(10 features) CSVs, aligns them by index, and returns tensors
suitable for ``MultimodalPDNet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Feature columns per modality (excludes id / label columns).
SPEECH_FEATURE_NAMES: list[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

HANDWRITING_FEATURE_NAMES: list[str] = [
    "mean_pressure", "pressure_variation", "mean_velocity",
    "velocity_variation", "mean_acceleration", "penup_time_ratio",
    "mean_stroke_length", "writing_tempo", "tremor_power",
    "fluency_score",
]

GAIT_FEATURE_NAMES: list[str] = [
    "stride_interval", "stride_variability", "swing_time",
    "stance_time", "double_support_time", "gait_speed",
    "cadence", "step_length", "stride_regularity",
    "gait_asymmetry",
]


class MultimodalPDDataset(Dataset):
    """Multimodal Parkinson's Disease dataset.

    Each sample contains speech, handwriting, and gait feature
    vectors plus a binary label (0 = healthy, 1 = PD).

    Args:
        speech_features: [N, 22] numpy array.
        handwriting_features: [N, 10] numpy array.
        gait_features: [N, 10] numpy array.
        labels: [N] numpy array of 0/1.
    """

    def __init__(
        self,
        speech_features: np.ndarray,
        handwriting_features: np.ndarray,
        gait_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        assert len(speech_features) == len(labels)
        assert len(handwriting_features) == len(labels)
        assert len(gait_features) == len(labels)

        self.speech = torch.tensor(speech_features, dtype=torch.float32)
        self.handwriting = torch.tensor(handwriting_features, dtype=torch.float32)
        self.gait = torch.tensor(gait_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "speech": self.speech[idx],
            "handwriting": self.handwriting[idx],
            "gait": self.gait[idx],
            "label": self.labels[idx],
        }


def load_modality_csv(
    path: str | Path,
    feature_columns: list[str],
    label_column: str = "status",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a single modality CSV and return (features, labels).

    Args:
        path: Path to the CSV file.
        feature_columns: Ordered list of feature column names.
        label_column: Name of the binary label column.

    Returns:
        features: [N, num_features] float64 array.
        labels: [N] int array.
    """
    df = pd.read_csv(path)
    features = df[feature_columns].values.astype(np.float64)
    labels = df[label_column].values.astype(np.int64)
    logger.info(
        "Loaded %s: %d samples, %d features, PD=%d, Healthy=%d",
        Path(path).name, len(df), features.shape[1],
        int(labels.sum()), int((labels == 0).sum()),
    )
    return features, labels


def load_all_modalities(
    data_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load speech, handwriting, and gait CSVs from *data_dir*.

    Expects the following paths under *data_dir*::

        speech/parkinsons.csv
        handwriting/handwriting_data.csv
        gait/gait_data.csv

    Samples are aligned by row index.  If modality sizes differ the
    minimum length is used (extra rows are truncated).

    Returns:
        speech_features: [N, 22]
        handwriting_features: [N, 10]
        gait_features: [N, 10]
        labels: [N]  (from speech CSV)
    """
    data_dir = Path(data_dir)

    speech_feats, speech_labels = load_modality_csv(
        data_dir / "speech" / "parkinsons.csv",
        SPEECH_FEATURE_NAMES,
    )
    hw_feats, _ = load_modality_csv(
        data_dir / "handwriting" / "handwriting_data.csv",
        HANDWRITING_FEATURE_NAMES,
    )
    gait_feats, _ = load_modality_csv(
        data_dir / "gait" / "gait_data.csv",
        GAIT_FEATURE_NAMES,
    )

    # Align by truncating to the smallest dataset
    n = min(len(speech_feats), len(hw_feats), len(gait_feats))
    if n < len(speech_feats):
        logger.warning(
            "Modality sizes differ; truncating to %d samples.", n,
        )

    return (
        speech_feats[:n],
        hw_feats[:n],
        gait_feats[:n],
        speech_labels[:n],
    )
