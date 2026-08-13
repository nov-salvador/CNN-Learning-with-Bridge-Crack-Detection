# 🧠 Bridge Crack Detection using Deep Learning

> An end-to-end computer vision project for bridge crack detection using **PyTorch**, **Transfer Learning (ResNet18)**, **FastAPI**, **TensorBoard**, **MLflow**, and **Docker**.

This repository documents my journey of learning **Deep Learning** and **Machine Learning Engineering** by building a real-world image classification application. Rather than focusing solely on model accuracy, the project follows an end-to-end machine learning workflow—from data preparation and model training to experiment tracking, inference, API development, and containerization.

---

## 🚀 Project Highlights

- ✅ Bridge crack image classification using **ResNet18 Transfer Learning**
- ✅ Data preprocessing and augmentation pipeline
- ✅ Class-weighted training to address dataset imbalance
- ✅ Model checkpointing and resume training
- ✅ Early stopping and learning rate scheduling
- ✅ Experiment tracking with **MLflow**
- ✅ Training visualization using **TensorBoard**
- ✅ Image inference pipeline
- ✅ REST API built with **FastAPI**
- ✅ **Docker containerization**
- ✅ End-to-end web application
- ✅ Frontend deployment with **Netlify**
- 🚧 GitHub Actions (CI/CD) *(Planned)*
- 🚧 Cloud deployment *(Planned)*

---

## 📖 Overview

Bridge crack detection is a binary image classification problem where the objective is to classify bridge surface images into one of two categories:

- 🟥 Crack
- 🟩 No Crack

This project demonstrates an end-to-end deep learning workflow by covering:

- Dataset preparation
- Data augmentation
- Class imbalance handling
- Transfer Learning with ResNet18
- Model training and evaluation
- Experiment tracking
- Model checkpointing
- Image inference
- REST API development with FastAPI
- Docker containerization

The goal is not only to build an accurate classifier but also to gain practical experience with the tools and workflows commonly used by Machine Learning Engineers.

---

## 🎯 Objectives

- Learn the fundamentals of Convolutional Neural Networks (CNNs)
- Understand Transfer Learning
- Build image classification models using PyTorch
- Handle class imbalance during model training
- Track experiments using MLflow and TensorBoard
- Implement model checkpointing and resume training
- Develop a production-style inference pipeline
- Deploy the trained model through a FastAPI REST API
- Containerize the application using Docker
- Learn MLOps fundamentals including Docker and CI/CD

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

### Experiment Tracking

- TensorBoard
- MLflow

### Backend API

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- React
- Vite
- JavaScript
- CSS

### Containerization

- Docker
- Docker Compose

### Deployment

- Netlify
- Render

### Development Tools

- Git
- GitHub
- VS Code

---

## 📂 Project Structure

```text
CNN-Learning-with-Bridge-Crack-Detection/
│
├── app/
│   ├── routes/                 # FastAPI endpoints
│   ├── schemas.py              # Response models
│   └── main.py                 # FastAPI application
│
├── configs/
│   └── config.py               # Project configuration
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── notebooks/                  # Learning notebooks
│
├── src/
│   ├── dataset/                # Dataset and DataLoader
│   ├── inference/              # Model inference
│   ├── models/                 # Model architectures
│   ├── train/                  # Training and evaluation
│   └── utils/                  # Checkpointing and logging
│
├── checkpoints/                # Saved model checkpoints
├── logs/                       # Training logs and TensorBoard files
├── outputs/                    # Generated outputs
├── mlruns/                     # Mlflow artifacts 
├── frontend/                   # React frontend application
│
├── mlflow.db
├── compose.yaml
├── compose.dev.yaml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 🔄 Machine Learning Pipeline

```text
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
Data Augmentation
    │
    ▼
Transfer Learning (ResNet18)
    │
    ▼
Model Training
    │
    ▼
Evaluation
    │
    ▼
Best Model Checkpoint
    │
    ▼
Image Inference
    │
    ▼
FastAPI REST API
    │
    ▼
Docker Container
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Upload an image for prediction |

---

## 📊 Experiment Tracking

The training process includes:

- TensorBoard for monitoring loss and accuracy
- MLflow for experiment tracking
- Model checkpointing
- Resume training support
- Learning rate scheduling

---

## 🐳 Docker

The application has been containerized using **Docker** to provide a reproducible environment for the machine learning application and API.

### Docker Components

The project includes:

- `Dockerfile` — Defines the Python environment and application image
- `compose.dev.yaml` — Defines and manages the application services
- Dockerized model training
- Dockerized MLflow
- Dockerized FastAPI application

### Docker Development

Docker Compose is used to manage the different services and development environment.

The project uses **bind mounts** for directories such as:

- `src/`
- `configs/`
- `data/`
- `checkpoints/`
- `outputs/`
- `mlruns/`

This allows changes to source code, configuration, data, and generated files to be reflected inside the running containers without rebuilding the Docker image.

### Build the Docker Image

```bash
docker compose -f compose.dev.yaml up --build
```

---

## 📈 Current Progress

### ✅ Completed

- Neural Networks Review
- CNN Fundamentals
- Dataset Preparation
- Data Augmentation
- Transfer Learning (ResNet18)
- Model Training
- Model Evaluation
- TensorBoard Integration
- MLflow Integration
- Image Inference Pipeline
- FastAPI REST API
- Docker Containerization
- React Frontend
- End-to-End Web Application

### 🚧 In Progress

- GitHub Actions (CI/CD)


### 📅 Planned
- Cloud Deployment


---

## 📚 Learning Outcomes

Through this project, I'm gaining hands-on experience with:

- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Computer Vision
- Experiment Tracking
- Model Deployment
- FastAPI
- REST APIs
- MLOps Fundamentals
- Docker
- CI/CD

---

## 🤝 Acknowledgements

Special thanks to **OpenAI's ChatGPT** for serving as an AI mentor throughout this learning journey by explaining deep learning concepts, reviewing code, discussing software engineering practices, and guiding the development of the complete machine learning pipeline.

---

## 📬 Connect With Me

**Nov Larsley Salvador**

Licensed Civil Engineer transitioning into **Machine Learning Engineering** and **Data Engineering**.

If you have suggestions, feedback, or would like to discuss Machine Learning, Computer Vision, Deep Learning, or MLOps, feel free to connect!

- **LinkedIn:** https://www.linkedin.com/in/nov-larsley-salvador-87b430224/

⭐ If you found this repository helpful, consider giving it a star!