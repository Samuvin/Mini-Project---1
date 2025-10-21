# Dataset Documentation

## Overview

This directory contains datasets for Parkinson's Disease detection across three modalities: speech, handwriting, and gait patterns.

## Data Sources

### 1. Speech Data: UCI Parkinson's Dataset

**Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/parkinsons
- **Size**: 195 samples (147 PD, 48 healthy)
- **Features**: 23 biomedical voice measurements

**Features Include**:
- MDVP:Fo(Hz) - Average vocal fundamental frequency
- MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
- MDVP:Flo(Hz) - Minimum vocal fundamental frequency
- MDVP:Jitter(%), Jitter(Abs), RAP, PPQ, DDP - Jitter measures
- MDVP:Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA - Shimmer measures
- NHR, HNR - Noise-to-harmonics ratios
- RPDE, DFA - Nonlinear complexity measures
- Spread1, Spread2, D2, PPE - Nonlinear dynamical measures

**Citation**:
```
'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection',
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM.
BioMedical Engineering OnLine 2007, 6:23
```

### 2. Handwriting Data: PaHaW Database

**Source**: Parkinson's Handwriting Database
- **URL**: https://wwwp.fc.upaep.mx/~minhchau/
- **Samples**: Handwriting samples from PD patients and healthy controls

**Features Include**:
- Pen pressure variations
- Writing velocity
- Acceleration patterns
- Pen-up time (time spent not writing)
- Stroke length and duration
- Writing fluency metrics

**Citation**:
```
PaHaW: Parkinson Handwriting Database
Orozco-Arroyave, J.R., et al.
```

### 3. Gait Data: PhysioNet Gait Database

**Source**: PhysioNet - Gait in Parkinson's Disease
- **URL**: https://physionet.org/content/gaitpdb/1.0.0/
- **Samples**: Gait recordings from PD patients and controls

**Features Include**:
- Stride interval (time between successive heel-strikes)
- Swing time (foot off ground)
- Stance time (foot on ground)
- Left-right stride time variability
- Gait rhythm measures

**Citation**:
```
Hausdorff JM, Lertratanakul A, Cudkowicz ME, Peterson AL, Kaliton D, Goldberger AL.
Dynamic markers of altered gait rhythm in amyotrophic lateral sclerosis.
Journal of Applied Physiology 2000;88:2045-2053.
```

## Directory Structure

```
data/
├── raw/
│   ├── speech/
│   │   └── parkinsons.csv
│   ├── handwriting/
│   │   └── handwriting_features.csv
│   └── gait/
│       └── gait_data.csv
├── processed/
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── y_test.npy
└── README.md
```

## Data Format

### Raw Data Format

All datasets should be in CSV format with:
- First row: column headers (feature names)
- Subsequent rows: samples
- Last column: target label (1 for PD, 0 for healthy)

### Processed Data Format

Processed data is saved as NumPy arrays (.npy):
- Features (X): Float arrays with normalized values
- Labels (y): Integer arrays (0 or 1)

## Data Download Instructions

### Automated Download

Run the data loader script:
```bash
python src/data/data_loader.py --download
```

### Manual Download

1. **Speech Data**:
   - Visit: https://archive.ics.uci.edu/ml/datasets/parkinsons
   - Download: parkinsons.data
   - Save as: `data/raw/speech/parkinsons.csv`

2. **Handwriting Data**:
   - Visit: PaHaW database website
   - Request access and download
   - Extract features and save as: `data/raw/handwriting/handwriting_features.csv`

3. **Gait Data**:
   - Visit: https://physionet.org/content/gaitpdb/1.0.0/
   - Download database
   - Process and save as: `data/raw/gait/gait_data.csv`

## Data Statistics

### Class Distribution

| Dataset | Total Samples | PD Cases | Healthy Controls | Class Ratio |
|---------|--------------|----------|------------------|-------------|
| Speech | ~195 | ~147 | ~48 | 3:1 (imbalanced) |
| Handwriting | Variable | Variable | Variable | Variable |
| Gait | Variable | Variable | Variable | Variable |

### Feature Statistics

- Speech: 22 acoustic features
- Handwriting: 10-15 kinematic features
- Gait: 8-12 temporal-spatial features
- Combined: 40-50 features after fusion

## Preprocessing Steps

1. **Missing Value Handling**: Forward fill or mean imputation
2. **Outlier Detection**: IQR method or Z-score
3. **Normalization**: StandardScaler (mean=0, std=1)
4. **Feature Selection**: Correlation analysis and feature importance
5. **Class Balancing**: SMOTE for minority class oversampling

## Privacy and Ethics

- All datasets are publicly available and anonymized
- No personally identifiable information (PII) is included
- Data is used solely for research and educational purposes
- Comply with data usage terms from original sources

## License

Each dataset has its own license terms. Please refer to the original sources for licensing information.

## Notes

- Ensure you have permission and proper citations when using these datasets
- Some datasets may require registration or data use agreements
- Always validate data integrity after download
- Check for updated versions of datasets periodically

