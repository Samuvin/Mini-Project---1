# Parkinson's Disease Datasets

This document describes the three real datasets used in this multimodal detection system.

## 1. Speech Data - UCI Parkinson's Dataset

### Overview
- **Dataset**: UCI Parkinson's Disease Dataset
- **Source**: https://archive.ics.uci.edu/ml/datasets/Parkinsons
- **Status**: ✅ Implemented and automatically downloaded

### Details
- **Samples**: 195 voice recordings (147 from PD patients, 48 healthy)
- **Features**: 22 acoustic measurements
- **Format**: CSV file (`parkinsons.csv`)

### Features List
1. MDVP:Fo(Hz) - Average vocal fundamental frequency
2. MDVP:Fhi(Hz) - Maximum vocal fundamental frequency  
3. MDVP:Flo(Hz) - Minimum vocal fundamental frequency
4. MDVP:Jitter(%) - Jitter percentage
5. MDVP:Jitter(Abs) - Absolute jitter in microseconds
6. MDVP:RAP - Relative amplitude perturbation
7. MDVP:PPQ - Five-point period perturbation quotient
8. Jitter:DDP - Average absolute difference of differences
9. MDVP:Shimmer - Shimmer
10. MDVP:Shimmer(dB) - Shimmer in decibels
11. Shimmer:APQ3 - Three-point amplitude perturbation quotient
12. Shimmer:APQ5 - Five-point amplitude perturbation quotient
13. MDVP:APQ - Amplitude perturbation quotient
14. Shimmer:DDA - Average absolute difference of differences
15. NHR - Noise-to-harmonics ratio
16. HNR - Harmonics-to-noise ratio
17. RPDE - Recurrence period density entropy
18. DFA - Detrended fluctuation analysis
19. spread1 - Nonlinear measure of fundamental frequency variation
20. spread2 - Nonlinear measure of fundamental frequency variation
21. D2 - Correlation dimension
22. PPE - Pitch period entropy

### Citation
```
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008),
'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease',
IEEE Transactions on Biomedical Engineering
```

### Installation
The system automatically downloads this dataset on first run. If download fails, manually download from:
https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data

Place the file at: `data/raw/speech/parkinsons.csv`

---

## 2. Handwriting Data - PaHaW Database

### Overview
- **Dataset**: PaHaW Parkinson's Disease Handwriting Database
- **Alternative**: NewHandPD Dataset
- **Source**: Research institutions (requires request)

### Details
- **Samples**: Handwriting from PD patients and healthy controls
- **Features**: Pressure, velocity, acceleration, pen trajectories, timing
- **Format**: Time-series data or extracted features (CSV)

### Typical Features (10-15 features)
1. mean_pressure - Average pen pressure
2. std_pressure - Pressure variation
3. mean_velocity - Average writing velocity
4. std_velocity - Velocity variation
5. mean_acceleration - Average acceleration
6. pen_up_time - Time pen lifted from paper
7. stroke_length - Average stroke length
8. writing_tempo - Overall writing speed
9. tremor_frequency - Frequency of tremor in writing
10. fluency_score - Writing fluency measure

### Public Alternatives
Since PaHaW requires institutional access, this system is configured to work with:

1. **NewHandPD** - Publicly available subset
2. **Extracted features** from research papers
3. **Sample data** for demonstration purposes

### Installation
Due to access restrictions, users need to:

1. **Option A**: Request access from PaHaW database maintainers
2. **Option B**: Use publicly available handwriting feature datasets
3. **Option C**: Extract features from digitizer pen data

**File location**: `data/raw/handwriting/handwriting_features.csv`

**Expected format**:
```csv
mean_pressure,std_pressure,mean_velocity,std_velocity,...,status
0.5,0.15,2.5,0.8,...,1
0.6,0.12,2.8,0.7,...,0
```

Where `status`: 1 = Parkinson's Disease, 0 = Healthy

---

## 3. Gait Data - PhysioNet Gait Database

### Overview
- **Dataset**: Gait in Parkinson's Disease Database
- **Source**: https://physionet.org/content/gaitpdb/1.0.0/
- **Status**: Publicly available for download

### Details
- **Samples**: Gait measurements from PD patients and controls
- **Features**: Stride intervals, timing parameters from force sensors
- **Format**: Text files with time-series stride data

### Typical Features (8-12 features)
1. stride_interval - Time between heel strikes
2. stride_interval_std - Variability in stride
3. swing_time - Time foot off ground
4. stance_time - Time foot on ground
5. double_support - Time both feet on ground
6. gait_speed - Walking velocity
7. cadence - Steps per minute
8. step_length - Distance per step
9. stride_regularity - Consistency of gait pattern
10. gait_asymmetry - Left-right differences

### Download Instructions

1. **Visit**: https://physionet.org/content/gaitpdb/1.0.0/
2. **Download**: Click "Download ZIP" button
3. **Extract**: Unzip to temporary location
4. **Process**: Run preprocessing script (provided)

### Installation

**Automated**:
```bash
python scripts/download_gait_data.py
```

**Manual**:
1. Download from PhysioNet
2. Extract files
3. Place processed features at: `data/raw/gait/gait_features.csv`

**Expected format**:
```csv
stride_interval,stride_interval_std,swing_time,stance_time,...,status
1.1,0.05,0.4,0.7,...,1
1.15,0.04,0.42,0.68,...,0
```

Where `status`: 1 = Parkinson's Disease, 0 = Healthy

### Citation
```
Hausdorff JM, Lertratanakul A, Cudkowicz ME, Peterson AL, Kaliton D, Goldberger AL.
Dynamic markers of altered gait rhythm in amyotrophic lateral sclerosis.
J Appl Physiol 88: 2045-2053, 2000.
```

---

## Combined Dataset

### Total Features
- Speech: 22 features
- Handwriting: 10 features
- Gait: 10 features
- **Total**: 42 features

### Data Alignment
The system aligns samples across modalities by:
1. Loading all three datasets
2. Taking minimum sample count across datasets
3. Concatenating features: [speech | handwriting | gait]

### Sample Counts
- Target: 195 samples (limited by speech dataset)
- If handwriting/gait have fewer samples, system uses minimum

---

## Important Notes

### No Synthetic Data
⚠️ **This system does NOT generate synthetic data.** All features must come from real datasets.

If a dataset is missing:
- System will raise a clear error
- Provide download instructions
- Will NOT create fake/synthetic data as fallback

### Data Privacy
All datasets used are:
- Publicly available or accessible via research request
- De-identified patient data
- Published in peer-reviewed research

### License & Attribution
When using this system, please cite:
1. Original dataset papers (listed above)
2. This implementation (if publishing results)

---

## Setup Checklist

- [x] Speech data: `data/raw/speech/parkinsons.csv`
- [ ] Handwriting data: `data/raw/handwriting/handwriting_features.csv`
- [ ] Gait data: `data/raw/gait/gait_features.csv`

Run `python train.py` to check dataset status and train models.

