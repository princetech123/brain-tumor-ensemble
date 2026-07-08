# Brain Tumor Ensemble - Hinglish Complete Guide 🧠

## Table of Contents
1. [Project Ka Overview (प्रोजेक्ट का परिचय)](#project-overview)
2. [Installation (स्थापना)](#installation)
3. [Step-by-Step Walkthrough (चरण दर चरण)](#walkthrough)
4. [Technologies Explanation (तकनीकों की व्याख्या)](#technologies)
5. [Components Ki Details (घटकों का विवरण)](#components)
6. [Troubleshooting (समस्या समाधान)](#troubleshooting)

---

## Project Ka Overview 📋 {#project-overview}

### Yeh Project Kya Hai?

Iss project ka main purpose hai **brain tumors ko detect karna aur classify karna** MRI images se using advanced Artificial Intelligence. 

**Simple words mein:**
- Doctor ko MRI scan dete ho
- AI system ushe analyze karta hai
- Batata hai ki tumor hai ya nahi aur kaunsa type ka hai
- Saath mein visual explanation bhi deta hai (graphs aur heatmaps)

### Project Ke Main Features:

1. **Multiple AI Models (कई AI मॉडल)** - ek ek model alag alag tareeke se image analyze karta hai
2. **Smart Fusion (स्मार्ट फ्यूजन)** - sabhi models ke results ko combine karke best answer deta hai
3. **Explainable AI (समझाने योग्य AI)** - AI batata hai ki unhone kya dekha jab decision liya
4. **Interactive Dashboard (डैशबोर्ड)** - ek pretty web interface jo use karna bhaut asaan hai

---

## Installation Guide (स्थापना गाइड) 📥 {#installation}

### Step 1: Environment Setup (माहौल तैयार करना)

```bash
# Pehle project ko download karo
git clone https://github.com/your-username/brain_tumor_ensemble.git
cd brain_tumor_ensemble

# Virtual environment banao (Python ke liye alag space)
python3 -m venv venv

# Environment ko activate karo (Windows alag command)
# Mac/Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate
```

**Virtual Environment Kya Hai?**
- Iska matlab ek alag "sandbox" banate hain computer mein
- Jab project ke liye libraries install karni hain, toh wo sirf iss sandbox mein jati hain
- Isse baki programs affected nahi hote

### Step 2: Dependencies Install Karo (जरूरी चीजें लगाएं)

```bash
# Requirements.txt se sab packages install hote hain
pip install -r requirements.txt
```

**Kya Install Hoga?**
- **PyTorch**: Deep Learning ke liye (brain models)
- **TorchVision**: Image processing ke liye
- **Flask**: Web server banane ke liye
- **OpenCV**: Image processing ke liye
- **NumPy/Pandas**: Data handling ke liye

### Step 3: SSL Certificate Fix (Agar Download Issue Ho)

Mac pe kabhi certificate error aaye:
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

---

## Step-by-Step Walkthrough (पूरी प्रक्रिया) 🚀 {#walkthrough}

### Option A: Synthetic Data Generate Karna (Test Ke Liye)

**Agar real data nahi ho, toh practice ke liye fake data bana sakte ho:**

```bash
python3 generate_synthetic_data.py
```

**Kya Hota Hai?**
- Normal brain images banati hai
- Tumor (imaginary) add karti hai practice ke liye
- `data/sample_mri/` folder mein save hoti hai

### Option B: Real Data Use Karna

**Agar Kaggle se real MRI images download karo:**
1. Go to: `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`
2. Dataset download karo
3. `data/sample_mri/` folder mein extract karo

**Folder Structure Kya Hona Chahiye:**
```
data/sample_mri/
├── train/              (Training data - 80%)
│   ├── tumor/
│   └── normal/
├── val/                (Validation data - 10%)
│   ├── tumor/
│   └── normal/
└── test/               (Testing data - 10%)
    ├── tumor/
    └── normal/
```

### Step-by-Step: Application Run Karna

#### **Step 1: Virtual Environment Activate Karo**
```bash
source venv/bin/activate
```

#### **Step 2: (Optional) Training Data Generate Karo**
```bash
python3 generate_synthetic_data.py
```

#### **Step 3: (Optional) Unit Tests Run Karo**
```bash
python3 -m unittest discover tests/
```
*Isse confirm hota hai ki sab components sahi kaam kar rahe hain*

#### **Step 4: (Optional) Models Ko Train Karo**
```bash
python3 src/training.py
```

**Yeh Command Kya Karti Hai?**
- Pehle backbones (ResNet, EfficientNet, ViT) ko train karta hai
- Phir fusion model ko train karta hai
- Checkpoints save karta hai `checkpoints/` mein
- Graphs aur metrics save karta hai `logs/` mein

**Time Kitna Lagega?**
- GPU par: 10-20 minutes
- CPU par: 1-2 ghante

#### **Step 5: Web Application Start Karo** 🌐
```bash
PYTHONPATH=/Users/princeyadav/.gemini/antigravity/scratch/brain_tumor_ensemble:$PYTHONPATH python3 src/web/app.py
```

**Output Hoga Kauch Aise:**
```
 * Running on http://127.0.0.1:5001
 * Debugger is active!
```

#### **Step 6: Browser Mein Open Karo**
Browser mein jaao: `http://localhost:5001`

---

## Application Ka Use Kaise Kare 💻 {#application-usage}

### Main Dashboard (मुख्य पृष्ठ)

**Jab app khol doge, toh ye features dikhenge:**

1. **File Upload Area** - Drag & drop karo apni MRI image
2. **Predict Button** - Click karo analysis ke liye
3. **Results Panel** - Diagnosis aur explanation dikhega

### Prediction Process (पूर्वानुमान प्रक्रिया)

```
Step 1: Image Upload → Step 2: Preprocessing → Step 3: Model Analysis 
→ Step 4: Results Display
```

**Kya Hota Har Step Mein:**

1. **Image Upload (छवि अपलोड)**
   - MRI scan image select karo
   - JPG, PNG format mein hona chahiye

2. **Preprocessing (पूर्व-प्रसंस्करण)**
   - Image resize hoti hai (128x128 pixels)
   - Noise remove hota hai (Gaussian Blur)
   - Contrast improve hota hai (CLAHE)

3. **Model Analysis (मॉडल विश्लेषण)**
   - ResNet50 - image analyze karta hai
   - EfficientNet-B3 - alag tareeke se dekh ta hai
   - ViT - transformer ke through dekh ta hai
   - Fusion - teeno ke results combine karta hai

4. **Results Display (परिणाम प्रदर्शन)**
   - Diagnosis (Tumor hai ya Nahi)
   - Confidence score (kitna sure AI hai)
   - Visual explanation (heatmaps)
   - Model-wise predictions

---

## Technologies Ka Explanation (तकनीकों की व्याख्या) 🔬 {#technologies}

### 1. PyTorch (पाइटोर्च) 🔥
**Kya Hai:** Deep learning library jisme models banate aur train karte hain

**Simple Example:**
```python
import torch
# Ek tensor banao (multi-dimensional array)
x = torch.tensor([1, 2, 3])
print(x)  # Output: tensor([1, 2, 3])
```

**Humari Project Mein Kya Karta Hai:**
- Neural networks define karna
- Models ko train karna
- Predictions lena

### 2. OpenCV (ओपनसीवी)
**Kya Hai:** Image processing library - images ko edit/analyze karne ke liye

**Humari Project Mein Use:**

#### a) **Gaussian Blur (गॉसियन ब्लर)**
```python
import cv2
# Noise remove karne ke liye image ko smooth karte hain
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.0)
```
**Matlab:** Jaise photo mein slight blur aate hain, waise image smooth hoti hai, noise remove hota hai

#### b) **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(image)
```
**Matlab:** Image ki brightness aur contrast improve hoti hai taaki details clearly dikhein

#### c) **Skull Stripping (खोपड़ी अलगाव)**
```python
# Brain ko skull se alag karte hain
# Otsu thresholding use karte hain
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```
**Matlab:** Image se sirf brain parts ko nikaalte hain, baaki ke skull etc ko remove karte hain

### 3. TorchVision (टॉर्चविजन)
**Kya Hai:** PyTorch ka image library - pre-trained models aur transforms

**Pre-trained Models Kya Hote Hain:**
- Ek model ek bade dataset (jaise ImageNet) par pehle se trained hota hai
- Fir hum isse apne specific task ke liye fine-tune karte hain
- Faida: Training time kam hota hai, results better aate hain

### 4. Flask (फ्लास्क)
**Kya Hai:** Python mein web app banane ka framework

**Humari Project Mein:**
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')  # "/" par jaane se home page khulega
def home():
    return render_template('index.html')
```

**Kya Karti Hai:**
- User interface provide karta hai
- File upload handle karta hai
- Results user ko show karta hai

### 5. HTML/CSS/JavaScript (एचटीएमएल/सीएसएस/जावास्क्रिप्ट)
**Kya Hai:** Website ka user interface

- **HTML**: Structure (layout)
- **CSS**: Styling (colors, design)
- **JavaScript**: Interactions (buttons click hona, drag & drop, etc)

---

## Components Ki Details (घटकों का विवरण) 🏗️ {#components}

### 1. Preprocessing Module (पूर्व-प्रसंस्करण मॉड्यूल)
**File:** `src/preprocessing.py`

```python
class MRIPreprocessor:
    # Image ko ready karta hai models ke liye
    
    def gaussian_blur(self, img):
        # Noise remove karta hai
        
    def apply_clahe(self, img):
        # Contrast improve karta hai
        
    def skull_strip(self, img):
        # Skull ko remove karta hai
```

**Steps:**
1. Image load karo
2. Resize karo (128x128)
3. Gaussian blur lagao
4. CLAHE apply karo
5. Normalize karo (values 0-1 ke beech)

### 2. Model Backbones (मॉडल की बेकबोन)
**File:** `src/models/backbones.py`

#### **ResNet50 (रेजनेट)**
```python
class ResNet50Backbone(nn.Module):
    # Deep CNN - 50 layers wala
    # Brain mein feature detection ke liye
```

**Kya Hai:** 50-layer ka Convolutional Neural Network
- Bottom se top tak layer by layer process karta hai
- Har layer kuch specific pattern detect karta hai

#### **EfficientNet-B3**
**Kya Hai:** Efficient aur powerful CNN
- ResNet se chhota but zyada powerful
- Mobile/edge devices ke liye bhi use hota hai

#### **Vision Transformer (ViT)**
**Kya Hai:** Transformer architecture - sequence of patches ke roop mein image dekh ta hai
```
Image → 16x16 patches mein divide karo
→ Har patch ko token banao
→ Transformer se process karo
→ Classification karo
```

**Matlab:** Image ko word sequence ki tarah treat karta hai

### 3. Fusion Model (फ्यूजन मॉडल)
**File:** `src/models/fusion.py`

```python
class EnsembleFusionModel(nn.Module):
    # Teeno models ke outputs ko combine karta hai
    
    def forward(self, features_resnet, features_effnet, features_vit):
        # Multi-Head Self-Attention apply karta hai
        # Final prediction deta hai
```

**Kaise Kaam Karta Hai:**
1. Teeno models se features lete hain
2. Same dimension mein convert karte hain
3. Self-Attention use karke zyada important features ko weight dete hain
4. Final classification karte hain

**Analogy:** Jaise 3 doctors apna apna opinion dein, phir consensus banao, waise

### 4. Explainability Module (व्याख्या मॉड्यूल)
**File:** `src/models/explainability.py`

#### **Grad-CAM (Gradient-weighted Class Activation Map)**
```python
class GradCAM:
    # CNN layers se heatmap generate karta hai
    # Batata hai ki model ne image ke konse parts ko important samjha
```

**Kya Dikhata Hai:**
- Red areas = Tumor-like features
- Blue areas = Normal features

#### **ViT Attention Maps**
```python
class ViTAttentionMap:
    # Transformer attention weights dikhata hai
    # Batata hai patches mein se konse saath kaam karte the
```

### 5. Training Module (प्रशिक्षण मॉड्यूल)
**File:** `src/training.py`

**Training Process:**
```
Stage 1: Backbones Ko Train Karna
├── ResNet50 ko frozen weights se initialize karo
├── Unfreeze only last layers ko
└── Fine-tune karo dataset par

Stage 2: Fusion Model Ko Train Karna
├── Sab backbones ko freeze karo
├── Sirf fusion network ko train karo
└── Best weights save karo
```

### 6. Evaluation Module (मूल्यांकन मॉड्यूल)
**File:** `src/evaluation.py`

**Metrics Calculate Karte Hain:**

- **Accuracy (सटीकता):** Kitne predictions sahi the?
  ```
  Accuracy = (Correct Predictions) / (Total Predictions)
  ```

- **Sensitivity (संवेदनशीलता):** Jinhe tumor tha, unhe detect kiya?
  ```
  Sensitivity = True Positives / (True Positives + False Negatives)
  ```

- **Specificity (विशिष्टता):** Jinhe tumor nahi tha, un mein false alarm nahi aaya?
  ```
  Specificity = True Negatives / (True Negatives + False Positives)
  ```

- **AUC (Area Under Curve):** Model ki overall performance
  ```
  0.5 = random guess
  1.0 = perfect
  ```

### 7. Web Application (वेब एप्लिकेशन)
**File:** `src/web/app.py`

**Routes (Pages):**
```python
@app.route('/')              # Home page
@app.route('/predict', methods=['POST'])  # Prediction
@app.route('/analytics')     # Stats aur graphs
@app.route('/about')         # About page
```

**Key Functions:**
```python
def load_all_models():
    # Sab 4 models load karta hai
    
def predict():
    # Image upload le ta hai
    # Preprocessing karta hai
    # Sab models se prediction le ta hai
    # Results format karta hai
    # JSON mein return karta hai
```

---

## Data Flow Diagram (डेटा प्रवाह) 📊

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER UPLOADS MRI IMAGE                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────────┐
          │  PREPROCESSING (preprocessing.py)  │
          │  ├─ Resize (128x128)              │
          │  ├─ Gaussian Blur                 │
          │  ├─ CLAHE (Contrast)              │
          │  └─ Normalize                     │
          └────────────┬───────────────────────┘
                       │
           ┌───────────┼───────────┐
           │           │           │
           ▼           ▼           ▼
      ┌────────┐  ┌────────────┐  ┌──────┐
      │ResNet50│  │EfficientNet│  │ViT   │
      │ Model  │  │-B3 Model   │  │Model │
      └───┬────┘  └──────┬─────┘  └──┬───┘
          │              │           │
          │    Features  │           │
          └──────┬───────┴───────┬───┘
                 │               │
                 ▼               │
          ┌──────────────────┐  │
          │ Grad-CAM         │◄─┘
          │ (Heatmap CNN)    │
          └──────────────────┘
                 │
                 │ Attention Rollout (ViT)
                 │
          ┌──────────────────┐
          │ Fusion Model     │
          │ (Combine Features)
          │ + Multi-Head     │
          │   Attention      │
          └────────┬─────────┘
                   │
                   ▼
        ┌─────────────────────────┐
        │ RESULTS + EXPLANATIONS  │
        │ ├─ Diagnosis            │
        │ ├─ Confidence %         │
        │ ├─ Heatmaps             │
        │ └─ Model-wise Votes     │
        └─────────────────────────┘
```

---

## Configuration File (कॉन्फ़िगरेशन फ़ाइल) ⚙️

**File:** `config/config.yaml`

```yaml
# Paths - Folders kahaán hain
paths:
  dataset_dir: "data/sample_mri"           # Data folder
  checkpoint_dir: "checkpoints"            # Trained models
  log_dir: "logs"                          # Training logs

# Data Settings - Data kaisa handle hoga
data:
  image_size: [128, 128]                   # Image size (chhota = tez, bada = detailed)
  batch_size: 16                           # Ek saath 16 images process karo
  train_split: 0.8                         # 80% training

# Preprocessing - Image ko kaise process karenge
preprocessing:
  clahe_clip_limit: 2.0                    # Contrast kitna increase karo
  gaussian_kernel_size: [5, 5]             # Blur kitna strong
  enable_skull_strip: false                # Skull remove karo ya nahi

# Training - Model ko kaise train karenge
training:
  backbone:
    epochs: 5                              # 5 baar poora data se train karo
    learning_rate: 0.0001                  # Kitna tezi se seekhoga
    early_stopping_patience: 3             # 3 epochs tak improvement na ho toh ruko
```

**Kya Matlab Hai:**

- **Epochs:** Poora data se 5 baar train karna
- **Learning Rate:** Weights ko kitna update karna
- **Batch Size:** 16 images ek saath process karna (memory aur speed ka balance)
- **Early Stopping:** Jab improvement nahi aa rahi toh training roko, waste na karo

---

## File Structure Samajho (फ़ाइल संरचना) 📂

```
brain_tumor_ensemble/
│
├── src/                        # Main code
│   ├── preprocessing.py        # Image cleaning
│   ├── training.py             # Model training
│   ├── evaluation.py           # Performance metrics
│   │
│   ├── models/                 # Neural networks
│   │   ├── backbones.py        # ResNet, EfficientNet, ViT
│   │   ├── fusion.py           # Combination model
│   │   └── explainability.py   # Heatmap generation
│   │
│   └── web/                    # Web application
│       ├── app.py              # Main Flask server
│       ├── templates/          # HTML pages
│       │   ├── index.html      # Home page
│       │   ├── result.html     # Prediction results
│       │   ├── analytics.html  # Statistics
│       │   └── about.html      # About page
│       └── static/             # Website files
│           ├── css/            # Styling
│           ├── js/             # Interactive features
│           └── uploads/        # User uploaded images
│
├── checkpoints/                # Trained models
│   ├── resnet50.pth
│   ├── efficientnet_b3.pth
│   ├── vit_b_16.pth
│   └── fusion.pth
│
├── data/                       # Dataset
│   └── sample_mri/
│       ├── train/              # Training images
│       ├── val/                # Validation images
│       └── test/               # Test images
│
├── logs/                       # Training logs
│   └── metrics.json            # Performance metrics
│
├── tests/                      # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_fusion.py
│
├── config/
│   └── config.yaml             # Settings
│
├── requirements.txt            # Package list
├── Dockerfile                  # Container config
├── README.md                   # English guide
└── HINGLISH_WALKTHROUGH.md     # Yeh file!
```

---

## Common Commands (आम कमांड्स) 💬

### Basic Commands:

```bash
# Virtual environment activate (हर बार jaruri)
source venv/bin/activate

# Web app start karo
PYTHONPATH=$(pwd):$PYTHONPATH python3 src/web/app.py

# Tests run karo
python3 -m unittest discover tests/

# Synthetic data generate karo
python3 generate_synthetic_data.py

# Training start karo (time-consuming)
python3 src/training.py

# Virtual environment se bahar niklo
deactivate
```

### Docker Commands (Agar Docker use karna ho):

```bash
# Build karo
docker build -t brain-tumor-ensemble .

# Run karo
docker-compose up

# Stop karo
docker-compose down
```

---

## Troubleshooting (समस्या समाधान) 🔧 {#troubleshooting}

### Problem 1: "ModuleNotFoundError: No module named 'src'"

**Reason:** PYTHONPATH set nahi hai

**Solution:**
```bash
# Pehle current folder path dekho
pwd
# Output: /Users/princeyadav/.gemini/antigravity/scratch/brain_tumor_ensemble

# Phir is command se run karo:
PYTHONPATH=/Users/princeyadav/.gemini/antigravity/scratch/brain_tumor_ensemble:$PYTHONPATH python3 src/web/app.py
```

### Problem 2: "SSL: CERTIFICATE_VERIFY_FAILED"

**Reason:** macOS par Python ka SSL certificate nahi hai

**Solution:**
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

### Problem 3: "CUDA out of memory"

**Reason:** GPU memory full hai

**Solution:**
```bash
# app.py mein change karo:
device = torch.device("cpu")  # GPU ki jagah CPU use karo
```

### Problem 4: "No such file or directory: 'config/config.yaml'"

**Reason:** Galat folder se command run ho raha hai

**Solution:**
```bash
# Project folder mein jao
cd /Users/princeyadav/.gemini/antigravity/scratch/brain_tumor_ensemble

# Phir run karo
python3 src/web/app.py
```

### Problem 5: Slow Performance

**Reason:** Large image size ya batch size

**Solution:** `config/config.yaml` mein change karo:
```yaml
data:
  image_size: [64, 64]      # 128 se 64 karo (tezi hogi)
  batch_size: 8             # 16 se 8 karo
```

---

## Step-by-Step Flow Diagram (पूरी प्रक्रिया) 🎯

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: SETUP (पहली बार)                                  │
│  ├─ Virtual environment banao                              │
│  ├─ Dependencies install karo (pip install -r requirements) │
│  └─ Config file check karo                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  STEP 2: DATA PREPARATION (डेटा तैयारी)                     │
│  ├─ Synthetic data generate karo (ya real data download)    │
│  ├─ Folder structure check karo                            │
│  └─ Data splits verify karo (train/val/test)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  STEP 3: MODEL TRAINING (मॉडल प्रशिक्षण) - OPTIONAL        │
│  ├─ Backbone models ko fine-tune karo                      │
│  ├─ Checkpoints save karo                                  │
│  └─ Fusion model train karo                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  STEP 4: WEB APPLICATION START (एप्लिकेशन शुरू)            │
│  ├─ python3 src/web/app.py run karo                        │
│  ├─ PYTHONPATH set karo                                    │
│  └─ Models load hoga (3-5 minutes)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  STEP 5: USE APPLICATION (एप्लिकेशन इस्तेमाल करो)          │
│  ├─ Browser mein http://localhost:5001 खोलो              │
│  ├─ MRI image drag & drop karo                            │
│  ├─ Predict button click karo                             │
│  └─ Results dekho with heatmaps                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
               🎉 DONE! 🎉
```

---

## Key Takeaways (मुख्य बातें) 📌

1. **Ye Project 3 AI Models Ko Combine Karta Hai:**
   - ResNet50 (Traditional CNN)
   - EfficientNet-B3 (Efficient CNN)
   - ViT (Transformer Model)

2. **Fusion Ke Through Best Prediction Milta Hai:**
   - Teeno models se consensus leta hai
   - Multi-Head Attention se important features ko weight deta hai

3. **Explainability Important Hai:**
   - Sirf diagnosis nahi, heatmaps bhi dikhate hain
   - Doctor ko samajh aata hai AI ne kya dekha

4. **Web Interface Bahut User-Friendly Hai:**
   - Drag & drop se upload kar sakte ho
   - Results tut hi dekh jaate ho
   - Mobile-responsive design

5. **Easy Customization:**
   - `config.yaml` se settings change kar sakte ho
   - Different models try kar sakte ho
   - Apne data par train kar sakte ho

---

## Advanced Usage (उन्नत उपयोग) 🚀

### Apna Data Pe Train Karna:

```bash
# 1. Data upload karo correct folder structure mein
# data/sample_mri/train/, data/sample_mri/val/, etc

# 2. Training start karo
python3 src/training.py

# 3. Logs check karo
tail -f logs/metrics.json
```

### Custom Model Weights Use Karna:

```python
# checkpoints/ folder mein apni .pth files put karo
# App automatically load kar lega
# Filename exact hona chahiye:
# - resnet50.pth
# - efficientnet_b3.pth
# - vit_b_16.pth
# - fusion.pth
```

### Docker Se Run Karna (Production):

```bash
docker build -t brain-tumor .
docker run -p 5001:5001 brain-tumor
```

---

## Glossary (शब्दकोश) 📖

| Term | Hindi | Matlab |
|------|-------|--------|
| Model | मॉडल | AI system jo predict karta hai |
| Feature | फीचर | Image ka koi important aspect |
| Backbone | बेकबोन | Main neural network architecture |
| Checkpoint | चेकपॉइंट | Saved model weights |
| Epoch | एपॉक | Ek baar poora dataset se training |
| Batch | बैच | Ek saath process karne wale images |
| Fine-tuning | फाइन-ट्यूनिंग | Pre-trained model ko adjust karna |
| Gradient | ग्रेडिएंट | Weights ko update karne ki direction |
| Loss | लॉस | Error measure - jitna low utna better |
| Accuracy | सटीकता | % of correct predictions |
| Heatmap | हीटमैप | Colored visualization of importance |
| Attention | अटेंशन | Important parts ko focus karna |
| Fusion | फ्यूजन | Multiple things ko combine karna |

---

## Resources (संसाधन) 🌐

- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **OpenCV Docs**: https://docs.opencv.org/
- **Flask Docs**: https://flask.palletsprojects.com/
- **Vision Transformer Paper**: https://arxiv.org/abs/2010.11929
- **Kaggle Dataset**: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## Contact & Support (संपर्क) 📧

Agar koi problem ho ya question pooches na:
- GitHub issues mein pocho
- Documentation padho
- Code comments dekho

---

**Happy Learning! 🎓**

Yeh project medical AI ka ek amazing example hai. Samajhne mein thoda time lagega par bilkul worth it hai!

**Ek baar se samajh nahi aaye toh:**
1. Dhire-dhire read karo
2. Ek ek component ko debug karo
3. Print statements add karo data flow dekh ne ke liye
4. Online tutorials dekhone se pehle locally samajhne ki koshish karo

**Remember:** "If it doesn't work, check the error message carefully. 90% time samjh aata hai kya issue hai!" 😊

---

*Last Updated: July 8, 2026*
*For Brain Tumor Ensemble Project*
