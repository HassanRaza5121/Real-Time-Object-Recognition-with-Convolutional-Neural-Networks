
# 🧠 Real-Time Object Recognition with Convolutional Neural Networks  

This repository contains a **real-time object recognition system** developed using **Convolutional Neural Networks (CNNs)**.  
The project focuses on building a **lightweight, efficient, and accurate recognition pipeline** capable of running on **consumer-grade hardware** in real-time.  

---

## 📘 Table of Contents  
- [Introduction](#introduction)  
- [Dataset Collection and Preprocessing](#dataset-collection-and-preprocessing)  
- [Model Design and Training Approaches](#model-design-and-training-approaches)  
- [Hyperparameter Tuning](#hyperparameter-tuning)  
- [Model Evaluation](#model-evaluation)  
- [Real-Time Application](#real-time-application)  
- [Discussion](#discussion)  
- [Conclusion](#conclusion)  
- [Tech Stack](#tech-stack)  
- [How to Run](#how-to-run)  
- [Results](#results)  
- [Future Improvements](#future-improvements)  
- [License](#license)  
- [Contact](#contact)  

---

## 🔍 Introduction  

This project explores the development of a **real-time object recognition system** using CNNs.  
With the increasing demand for **on-device intelligence**, the system aims to balance accuracy, efficiency, and computational cost.  

Three different CNN-based approaches were explored:  
1. **Baseline CNN** — trained from scratch  
2. **Data-Augmented CNN** — improved via augmentation techniques  
3. **Transfer Learning** — fine-tuned ResNet18/MobileNetV2 models  

The best-performing model was integrated into a live **OpenCV + Streamlit** application for real-time detection.

---

## 🧩 Dataset Collection and Preprocessing  

- A **custom dataset** was manually created using **OpenCV** (4 object categories, 35–70 images each).  
- Data cleaning included removing noisy or irrelevant samples.  
- Split ratio: **20:5:10** for train, validation, and test sets.  
- Applied **image augmentation** (rotation, flipping, brightness shifts).  
- Preprocessing included **resizing** and **normalization** for efficient model convergence.  

---

## 🧠 Model Design and Training Approaches  

Implemented three CNN architectures for comparative evaluation:  
- **Baseline CNN:** A shallow model built from scratch.  
- **Data-Augmented CNN:** Improved model with augmented data to reduce overfitting.  
- **Transfer Learning:** Fine-tuned **ResNet18 / MobileNetV2** pretrained on ImageNet.  

All models were trained using **PyTorch** with:  
- Loss: Cross-Entropy  
- Optimizers: Adam / SGD (with momentum)  
- Training technique: Mini-batch gradient descent  

---

## ⚙️ Hyperparameter Tuning  

| Parameter | Range Tested | Best Value | Notes |
|------------|---------------|-------------|-------|
| Learning Rate | 0.001 – 0.01 | 0.001 | Stable convergence with Adam |
| Batch Size | 16 – 64 | 32 | Balanced memory & stability |
| Optimizer | Adam, SGD | Adam | Best for this dataset |
| Dropout | 0.2 – 0.5 | 0.3 | Prevents overfitting |

Transfer learning showed **highest accuracy** and best generalization performance while keeping computational cost low.

---

## 📊 Model Evaluation  

Models were evaluated on accuracy, precision, recall, and confusion matrices.  

| Model | Accuracy | Key Insights |
|--------|-----------|--------------|
| Baseline CNN | Moderate | Overfit small dataset |
| Augmented CNN | Improved recall | Better class balance |
| Transfer Learning | Highest | Most efficient trade-off |

---

## 🎥 Real-Time Application  

The best-performing model was deployed in a **real-time OpenCV application** integrated with **Streamlit**.  

- Captures webcam frames  
- Preprocesses and sends them to CNN  
- Displays predictions live on-screen with minimal latency  

✅ Achieved stable, **real-time detection** across multiple lighting and background conditions.  

---

## 💬 Discussion  

- **From-scratch CNNs** allowed architectural flexibility but suffered from data limitations.  
- **Augmentation** improved robustness but required more compute.  
- **Transfer Learning** was the most efficient and accurate, leveraging pre-trained knowledge.  
- Dataset challenges included inconsistent lighting and background noise.  

---

## 🏁 Conclusion  

This project successfully demonstrates **real-time object recognition using CNNs**.  
Key takeaways:  
- Transfer learning is optimal for small custom datasets.  
- Data augmentation boosts generalization.  
- Efficient model deployment is achievable on consumer hardware.  

Future directions include:  
- Expanding dataset diversity  
- Integrating **lightweight transformer-based models**  
- Optimizing for **edge devices (Jetson, Raspberry Pi)**  

---

## 🧰 Tech Stack  

`Python` • `PyTorch` • `TensorFlow` • `OpenCV` • `Streamlit` • `NumPy` • `Pandas`

---

## 🧑‍💻 How to Run  

```bash
# Clone the repository
git clone https://github.com/HassanRaza5121/RealTime-Object-Recognition.git

# Navigate into the project folder
cd RealTime-Object-Recognition

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
