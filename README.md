# 🌿 AgroAI - Plant Disease Detection System

AgroAI is an AI-powered web application that detects plant diseases from leaf images using Deep Learning (CNN) and TensorFlow.

This system helps farmers identify crop diseases early and take preventive action.

---

## 🚀 Project Overview

AgroAI allows users to:

- 📸 Upload a leaf image
- 🔍 Detect plant disease
- 📊 View confidence score
- 💡 Get recommended treatment

The model is trained on selected classes from the PlantVillage dataset.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- ImageDataGenerator

---

## 📂 Dataset Used

**PlantVillage Dataset**

Selected Classes:

- Tomato___Early_blight
- Tomato___Late_blight
- Potato___Healthy
- Potato___Early_blight

---

## 🏗 Model Architecture

The Convolutional Neural Network (CNN) consists of:

- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- Flatten Layer
- Dense Layer
- Dropout
- Softmax Output Layer

---

## 📊 Model Performance

- Training Accuracy: ~89%
- Validation Accuracy: ~90%
- Epochs: 20
- Image Size: 128x128

---

## 📁 Project Structure
Plant-Disease-Detection/
│
├── dataset/
├── train.py
├── app.py
├── plant_disease_model.h5
├── class_indices.json
├── requirements.txt
└── README.md


---

## 🖥 How To Run The Project

### 1️⃣ Clone the Repository
git clone https://github.com/VarshithaJ07/Plant_Disease_Detection.git
cd your-repo-name

---

### 2️⃣ Create Virtual Environment
python -m venv venv

Activate:

**Windows**:venv\Scripts\activate


**Mac/Linux**:source venv/bin/activate

---

### 3️⃣ Install Dependencies
pip install tensorflow keras opencv-python numpy streamlit scikit-learn


---

### 4️⃣ Train the Model
python train.py

This will generate:
- plant_disease_model.h5
- class_indices.json

---

### 5️⃣ Run the Streamlit Application
streamlit run app.py

The app will open in your browser.

---

## 🌱 Application Features

- Upload leaf image
- Real-time prediction
- Confidence percentage
- Recommended treatment
- Clean and modern UI

---

## 🔮 Future Improvements

- Transfer Learning (MobileNetV2)
- Multi-language support (e.g., Kannada)
- Real-time camera detection
- Deployment on Streamlit Cloud
- Mobile app integration

---

## 👩‍💻 Author

Varshitha J 
Engineering Student  
AI & Machine Learning Enthusiast  

---

