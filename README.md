# 🩺 SugarCare: Diabetes Companion

> **AI-powered diabetes management platform** — Comprehensive prediction and detection system for diabetes care

A web application developed as a Major Project at the Institute of Engineering, Tribhuvan University. SugarCare combines machine learning and deep learning to assist in diabetes prediction, diabetic foot ulcer detection, and diabetic retinopathy screening.

---

## 📸 Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/8043b6ca-e227-4c95-9101-a61e15a3740b" alt="Dashboard" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/ae429843-fd69-4343-9ef7-a8fb36b1aa6f" alt="Diabetes Prediction" width="400"/>
  <img src="https://github.com/user-attachments/assets/05e84b4d-a578-4f81-a0d8-71108e8782ab" alt="Foot Ulcer Detection" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cfda566d-ebf8-46f7-b40c-85fc499447c3" alt="Retinopathy Detection" width="400"/>
  <img src="https://github.com/user-attachments/assets/98dafe5b-7a5e-462f-9fc4-c752d3d28c3b" alt="Yoga Exercises" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/a874b359-c8e6-40fd-a8f4-c740a28bf8c2" alt="User Interface" width="400"/>
</p>

---

## ✨ Features

### 🔬 Diabetes Prediction System
- **SVM-based prediction** using clinical parameters:
  - Age, Gender, Hypertension, Heart Disease
  - Smoking Habits, BMI, HbA1c Level, Blood Glucose Level
- Accurate risk assessment for diabetes

### 🦶 Diabetic Foot Ulcer Detection
- **CNN-powered image classification**
- **94% accuracy** in detecting foot ulcers
- Binary classification: Ulcer vs Non-ulcer

### 👁️ Diabetic Retinopathy Detection
- **Advanced CNN model** for retinal image analysis
- **80% accuracy** in severity classification
- Stages: No DR → Mild → Moderate → Severe → Proliferative DR

### 🧘 Wellness Features
- 3D yoga exercise animations (Blender-rendered)
- Comprehensive diabetes management guidance

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Backend & ML algorithms |
| **Django** | Web framework |
| **NumPy** | Numerical computations |
| **Pandas** | Data preprocessing |
| **Scikit-Learn** | SVM implementation |
| **TensorFlow** | CNN training (foot ulcer) |
| **PyTorch** | CNN training (retinopathy) |
| **Keras** | Neural network design |
| **OpenCV** | Image preprocessing |
| **PIL (Pillow)** | Medical image handling |
| **Bootstrap** | Responsive UI design |
| **Blender** | 3D yoga animations |

---

## 🚀 Installation & Setup

### 1. Create Virtual Environment

```bash
# Install virtualenv
pip install virtualenv

# Create virtual environment
python -m venv hamro_environment

# Activate virtual environment
# Windows
hamro_environment\Scripts\activate
# Linux/macOS
source hamro_environment/bin/activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install django
pip install django-crispy-forms
pip install django-bootstrap4
pip install crispy-bootstrap4

# ML/DL libraries
pip install scikit-learn
pip install tensorflow
pip install torch torchvision
pip install keras
pip install opencv-python

# Data processing
pip install numpy
pip install pandas
pip install matplotlib
pip install Pillow
```

### 3. Run the Application

```bash
# Start Django development server
python manage.py runserver

# Open in browser
# http://127.0.0.1:8000/
```

---

## 📊 Model Performance

| Model | Accuracy | Purpose |
|-------|----------|---------|
| **SVM** | High | Diabetes risk prediction |
| **CNN (Foot Ulcer)** | 94% | Ulcer detection |
| **CNN (Retinopathy)** | 80% | Retinopathy staging |

---

## 📄 Documentation

### Research Paper & Reports
- [Research Paper](https://kecktmnepal-my.sharepoint.com/:b:/r/personal/deptcomp_kecktm_edu_np/Documents/Major%20Project%20BCT%202077%20Batch/DiabetesPredictionSystemUsingSVMResearchPaper.pdf?csf=1&web=1&e=clhLCc)
- [Project Report](https://kecktmnepal-my.sharepoint.com/:b:/r/personal/deptcomp_kecktm_edu_np/Documents/Major%20Project%20BCT%202077%20Batch/DiabetesPredictionSystemUsingSVM.pdf?csf=1&web=1&e=q3gYVn)

---

## 👥 Team Members

**Kathmandu Engineering College — BCT 077 Batch**

- **Deephang Thegim** — [@mrdeephang](https://github.com/mrdeephang)
- **Dipesh Awasthi**
- **Esparsh Tamrakar**
- **Pankaj Karki**

---

## 🎓 Project Context

**Institution:** Institute of Engineering, Tribhuvan University  
**College:** Kathmandu Engineering College  
**Project Type:** Major Project  
**Batch:** BCT 077 (2077 Batch)  
**Version:** 1.0.0

---

## 📄 License

All Rights Reserved © 2025 SugarCare Team
