# 🌾 Crop Disease Detection using CNN & ResNet (Transfer Learning)

This project builds a **Crop Disease Detection System** that classifies images of **maize and other plants** using deep learning.  
It uses both a **custom Convolutional Neural Network (CNN)** and a **ResNet18 transfer learning model** to detect plant leaf diseases from the **PlantVillage dataset**.

---

## 📘 Project Overview

Plant diseases can drastically reduce crop yield and threaten food security.  
This project aims to develop a reliable model that can automatically identify plant diseases from leaf images.

We compare two approaches:
- 🧠 A **custom CNN** built from scratch  
- ⚡ A **ResNet18 model** using **transfer learning** for better accuracy and efficiency

---

## 🧩 Dataset

- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Classes:** Includes multiple crop disease categories such as:
Pepper__bell___Bacterial_spot
Tomato_Late_blight
Tomato_Leaf_Mold
Potato_Early_blight
Tomato_YellowLeaf__Curl_Virus
... (etc.)

- **Total Samples:** ~50,000+ labeled leaf images  
- **Split:**  
- 80% → Training  
- 20% → Validation

---

## ⚙️ Project Structure
├── notebooks/
│ ├── plant_disease_detection_cnn_resnet.ipynb # Main training notebook
├── models/
│ ├── plant_disease_cnn.pth # Trained CNN model weights
│ ├── plant_disease_resnet.pth # Trained ResNet model weights
├── data/
│ └── plantvillage_dataset.zip # (optional) dataset storage
├── README.md


---

## 🚀 How to Run

1️⃣ Open in Google Colab
You can run this project easily in **Google Colab** with GPU enabled.

2️⃣ Train the CNN
model = PlantDiseaseCNN()
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train for 5 epochs

3️⃣ Train the ResNet (Transfer Learning)
from torchvision import models
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Freeze feature layers, fine-tune final FC layer

4️⃣ Evaluate and Visualize
plt.plot(train_losses, label='Training Loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training Progress')

🧠 Key Concepts Learned
Image preprocessing with PyTorch Transforms
Dataset management using ImageFolder and DataLoader
Designing and training a CNN from scratch
Implementing transfer learning with ResNet
Evaluating model performance using accuracy curves

💡 Future Improvements
Add a web interface for live leaf disease prediction
Train on maize-specific data subset for specialized accuracy
Experiment with EfficientNet or Vision Transformers (ViT)
Optimize training using mixed precision (AMP)

🧑🏽‍💻 Author
Kwadwo
Student, Academic City University — Artificial Intelligence Major
📍 Ghana


