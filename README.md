MRI Brain Tumor Classification

This project focuses on **classifying brain MRI scans** into four categories — **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary** using both **custom-built CNN architectures** and **pre-trained deep learning models**.

The goal is to:
- Design and evaluate a **Custom CNN** from scratch as a baseline.
- Compare its performance with **transfer learning models** such as **VGG16**, **MobileNet**, and **ResNet**.
- Analyze how architectural complexity and pretraining impact classification accuracy and generalization, especially on a **small grayscale medical dataset**.


## Dataset
The dataset used in this project is publicly available on Kaggle:  
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

It contains **T1-weighted contrast-enhanced MRI images** grouped into four folders:
- `glioma_tumor/`
- `meningioma_tumor/`
- `no_tumor/`
- `pituitary_tumor/`

A smaller subset (~600 images) is used here to simulate limited-data medical scenarios.

## Project Outline
1. **Data Understanding & Preprocessing**
   - Image resizing, grayscale conversion, and normalization  
   - Mild augmentation (rotation, shift, zoom)  
   - Train/Validation/Test split  

2. **Model Development**
   - Implement a **Custom CNN** (LeakyReLU, BatchNorm, Dropout)  
   - Train and evaluate **VGG16**, **MobileNet**, and **ResNet** for comparison  
   - Apply **early stopping** and **learning rate scheduling** for stable optimization  

3. **Evaluation & Analysis**
   - Compare metrics (accuracy, F1-score, confusion matrix) across models  
   - Study overfitting trends and generalization on small medical datasets  