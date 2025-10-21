# Enhanced Prediction Features - User Guide

## 🎤 Audio Input (NEW!)

### Option 1: Record Audio via Microphone
1. Go to `http://localhost:5000/predict_page`
2. Click the **"Audio"** tab
3. Click the red **"Record Audio"** button
4. Allow microphone access when prompted
5. Speak clearly for 3-5 seconds (e.g., say "Ahhh" or read a sentence)
6. Click **"Stop Recording"**
7. Audio will be automatically processed and features extracted

### Option 2: Upload Audio File
1. Go to the **"Audio"** tab
2. Click **"Upload Audio File"**
3. Select a .wav, .mp3, or .m4a file
4. File will be automatically processed

**Supported Formats:** WAV, MP3, M4A (Max 10MB)

---

## ✍️ Handwriting Input (NEW!)

### Upload Handwriting Image
1. Click the **"Handwriting"** tab
2. Click **"Upload Handwriting Image"**
3. Select a photo of:
   - Your signature
   - A sentence you wrote
   - A spiral drawing
   - Any handwriting sample

**Tips for Best Results:**
- Use good lighting
- Write on white paper
- Take a clear, focused photo
- Keep the paper flat (no shadows or folds)

**Supported Formats:** JPG, JPEG, PNG, GIF

---

## 📋 Manual/CSV Input (Original)

### Option 1: Manual Entry
1. Click the **"Manual/CSV"** tab
2. Select **"Manual Entry"**
3. Enter all 42 feature values separated by commas
4. Click **"Make Prediction"**

### Option 2: Upload CSV
1. Select **"Upload CSV"**
2. Choose a CSV file with features
3. First row should be headers
4. Click **"Make Prediction"**

### Option 3: Example Data
1. Select **"Example Data"**
2. Pre-computed features will be loaded
3. Click **"Make Prediction"** to test

---

## 🎯 How It Works

### Feature Extraction Pipeline:

```
1. AUDIO INPUT
   Audio Recording/File → Feature Extraction → 22 Speech Features
   - MFCC coefficients
   - Jitter (frequency variation)
   - Shimmer (amplitude variation)  
   - Pitch, energy, HNR

2. HANDWRITING INPUT
   Image Upload → Feature Extraction → 10 Handwriting Features
   - Pressure variations
   - Velocity and acceleration
   - Tremor frequency
   - Pen-up time
   - Fluency scores

3. GAIT DATA (Auto-generated for demo)
   Default Values → 10 Gait Features
   - Stride length
   - Cadence
   - Asymmetry
   - Regularity

4. COMBINATION
   22 Speech + 10 Handwriting + 10 Gait = 42 Total Features

5. PREDICTION
   42 Features → SVM Model → Prediction + Confidence Score
```

---

## 🚀 Quick Start Guide

### To Test the Enhanced System:

1. **Open the app:** `http://localhost:5000/predict_page`

2. **Record your voice:**
   - Click "Audio" tab
   - Click red "Record Audio" button
   - Allow microphone access
   - Speak for 3-5 seconds
   - Click "Stop Recording"
   - ✅ See "Audio processed! 22 speech features extracted"

3. **Upload handwriting:**
   - Click "Handwriting" tab
   - Upload a photo of your handwriting
   - ✅ See "Image processed! 10 handwriting features extracted"

4. **Make prediction:**
   - Features are automatically combined
   - Click "Make Prediction" in the Manual/CSV tab
   - ✅ See results with confidence scores!

---

## 🎨 UI Features

### Modern Tabbed Interface:
- **🎤 Audio Tab:** Microphone recording + file upload
- **✍️ Handwriting Tab:** Image upload with preview
- **⌨️ Manual/CSV Tab:** Traditional feature input

### Visual Feedback:
- ✅ Green checkmarks when features extracted
- 📊 Progress indicators during processing
- 🖼️ Image preview for handwriting
- 📢 Status notifications

### Smart Feature Combination:
- Automatically combines all modalities
- Shows feature count for each modality
- Displays total features ready for prediction
- Validates that you have all 42 features

---

## 🔧 Technical Details

### Browser Requirements:
- **Microphone Recording:** Chrome, Firefox, Edge, Safari (requires HTTPS in production)
- **File Upload:** All modern browsers
- **Microphone Permission:** Must be granted by user

### API Endpoints:
- `POST /api/process_audio` - Extract speech features from audio
- `POST /api/process_handwriting` - Extract features from handwriting image
- `POST /api/predict` - Make prediction with combined features

### Feature Extraction:
- **Audio:** Uses librosa for speech analysis (currently simulated)
- **Handwriting:** Image processing and kinematic analysis (currently simulated)
- **Gait:** Auto-generated default values for demonstration

---

## ⚠️ Important Notes

1. **Demo Mode:** Feature extraction is currently simulated with realistic values. In production, actual signal processing would be implemented.

2. **Microphone Access:** Browser will ask for permission to access your microphone. This is required for recording.

3. **Privacy:** Audio and images are processed server-side and immediately deleted after feature extraction.

4. **File Sizes:** 
   - Audio: Max 10MB
   - Images: Max 5MB recommended

5. **For Clinical Use:** This is a research tool. Always consult healthcare professionals for diagnosis.

---

## 🎬 Try It Now!

**Refresh your browser at `http://localhost:5000/predict_page` and test the new features!**

The enhanced UI with audio recording and handwriting upload is now live! 🚀

