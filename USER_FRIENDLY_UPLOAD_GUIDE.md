# User-Friendly File Upload System

## 🎉 New Features - No Technical Knowledge Required!

The Parkinson's Disease Detection System now supports **automatic feature extraction** from uploaded files. Users no longer need to understand technical features - just upload your files!

---

## 📤 How to Use

### 1. Speech/Audio Analysis

**Option A: Record Your Voice**
1. Click "Start Recording" button
2. Say "Aaaaahhh" steadily for 3-5 seconds
3. Click "Stop Recording"
4. System automatically extracts 22 speech features

**Option B: Upload Audio File**
1. Click "Choose File" and select an audio file
2. Supported formats: MP3, WAV, OGG, M4A, FLAC
3. Click "Extract Features from Audio"
4. System processes the audio and extracts features automatically

**What it analyzes:**
- Vocal frequency (pitch)
- Voice variation (jitter)
- Voice amplitude variation (shimmer)
- Harmonics-to-noise ratio
- Nonlinear complexity measures

---

### 2. Handwriting Analysis

**Instructions:**
1. Draw a spiral on white paper OR write a sentence
2. Take a clear photo with good lighting
3. Upload the image (JPG, PNG, etc.)
4. Click "Extract Features from Image"

**What it analyzes:**
- Stroke thickness (pressure estimate)
- Writing smoothness
- Tremor frequency
- Writing tempo
- Fluency score

**Note:** Best results with high-contrast images (dark writing on white paper)

---

### 3. Gait/Walking Analysis

**Instructions:**
1. Record a video of yourself walking
2. Walk naturally for 10-15 seconds
3. Side view works best
4. Upload the video (MP4, MOV, etc.)
5. Click "Extract Features from Video"

**What it analyzes:**
- Step frequency (cadence)
- Stride length
- Walking speed
- Balance and regularity
- Left/right asymmetry

**Tips:**
- Good lighting helps
- Clear background recommended
- Normal walking pace

---

## 🎯 Making a Prediction

### You can use:
- ✅ Just speech (audio/recording)
- ✅ Just handwriting (image)
- ✅ Just gait (video)
- ✅ Any combination of the above!

**More modalities = Better accuracy**

### Steps:
1. Upload file(s) from any tab
2. Wait for "Success!" message
3. Click "Make Prediction" button
4. View results on the right side

---

## 📊 Understanding Results

### Prediction Result
- **"Healthy"** = Low probability of Parkinson's
- **"Parkinson's Disease Detected"** = High probability

### Confidence Score
- **80-100%** = High confidence (green)
- **60-80%** = Medium confidence (blue)
- **Below 60%** = Lower confidence (yellow)

### Probability Distribution
- Shows exact percentage for each outcome
- Adds up to 100%

---

## 🔬 Technical Details

### Feature Extraction Technology

**Audio Processing:**
- Library: `librosa` + `praat-parselmouth`
- Extracts: 22 acoustic features
- Matches UCI Parkinson's dataset format
- Includes: F0, jitter, shimmer, HNR, RPDE, DFA, etc.

**Image Processing:**
- Library: `OpenCV` + `scikit-image`
- Extracts: 10 handwriting features
- Analyzes: stroke patterns, tremor, pressure, fluency
- Note: Estimates from static images (real digitizers are more accurate)

**Video Processing:**
- Library: `OpenCV`
- Extracts: 10 gait features
- Analyzes: motion patterns, step detection, walking characteristics
- Note: Estimates from video (professional gait labs use force plates)

### Machine Learning Model
- Algorithm: Support Vector Machine (SVM)
- Kernel: RBF (optimized)
- Training: Real patient data only (no synthetic data)
- Validation: 5-fold cross-validation
- Accuracy: ~90-93%

---

## 📁 File Requirements

### Audio Files
- **Max size:** 50MB
- **Formats:** WAV, MP3, OGG, M4A, FLAC
- **Recommendation:** 3-5 seconds of sustained vowel sound

### Image Files
- **Max size:** 50MB
- **Formats:** JPG, JPEG, PNG, BMP, TIFF
- **Recommendation:** High contrast, clear photo, good lighting

### Video Files
- **Max size:** 50MB
- **Formats:** MP4, AVI, MOV, MKV
- **Recommendation:** 10-15 seconds, side view, normal walking

---

## ⚠️ Important Notes

### Accuracy Considerations
1. **Audio:** Most accurate when matching UCI dataset recording protocol
2. **Handwriting:** Static images are less accurate than real digitizer data
3. **Gait:** Video analysis is less accurate than professional motion capture

### For Best Results
- Use multiple modalities together
- Ensure good quality recordings/images/videos
- Follow the instructions carefully
- Use consistent lighting and environment

### Medical Disclaimer
**This system is for research and educational purposes only.**
- Always consult healthcare professionals for medical diagnosis
- This is NOT a replacement for clinical examination
- Results should be interpreted by qualified medical personnel

---

## 🔧 API Endpoints (For Developers)

```bash
# Test upload API
curl http://localhost:8000/api/upload/test

# Upload audio file
curl -X POST -F "file=@audio.wav" http://localhost:8000/api/upload/audio

# Upload handwriting image
curl -X POST -F "file=@handwriting.jpg" http://localhost:8000/api/upload/handwriting

# Upload gait video
curl -X POST -F "file=@walking.mp4" http://localhost:8000/api/upload/gait

# Make prediction with extracted features
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"speech_features": [119.99, ...], "handwriting_features": [0.45, ...], "gait_features": [1.15, ...]}'
```

---

## 📞 Support

**Server running at:** http://localhost:8000

**Access the system:**
1. Open browser
2. Go to http://localhost:8000
3. Click "Start Prediction"
4. Upload your files!

---

## 🎓 Educational Value

This system demonstrates:
- Machine Learning for healthcare
- Multimodal data fusion
- Real-time feature extraction
- Production-ready ML deployment
- Biomedical signal processing

Perfect for:
- Research projects
- Educational demonstrations
- ML/AI portfolio projects
- Healthcare technology learning

---

## 🚀 Future Enhancements

Potential improvements:
- Real-time video processing with pose estimation
- Mobile app integration
- Time-series handwriting capture (digitizer support)
- Historical tracking dashboard
- PDF report generation
- Multi-language support

---

**Built with:**
- Flask (Web Framework)
- Gunicorn (Production Server)
- librosa (Audio Processing)
- OpenCV (Image/Video Processing)
- scikit-learn (Machine Learning)
- Bootstrap (UI/UX)

**Version:** 2.0 - File Upload Edition
**Date:** October 2025

