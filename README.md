## MRI Brain Tumor Classification

This project focuses on classifying brain MRI scans into four categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary**, using a **ResNet50 based transfer learning model**.

The goal is to fine tune a pretrained ResNet50 model on a medical imaging dataset to achieve high classification accuracy while maintaining efficient training and strong generalization on unseen data.

---

## Dataset

The dataset used in this project is publicly available on Kaggle:  
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

It contains **T1 weighted contrast-enhanced MRI scans** categorized into four folders:
- `glioma_tumor/`
- `meningioma_tumor/`
- `no_tumor/`
- `pituitary_tumor/`

The dataset was used in its **original structured format**, with:
- `data/Training/` – used for model training and validation (via internal split)
- `data/Testing/` – used for final evaluation

---

## Project Outline

### 1. Data Understanding
- Explored dataset structure, image dimensions, and class distribution.  
- Verified grayscale consistency and relative class balance across tumor categories.  
- Visualized sample MRI images for each class to understand texture and boundary variations.  
- Confirmed that all classes were properly represented before model training.

### 2. Model Development
- Implemented a **ResNet50 classifier** with pretrained ImageNet weights as a feature extractor.  
- Added a custom classification head including:
  - Global Average Pooling  
  - Batch Normalization and Dropout layers  
  - Dense layer with softmax activation for four class prediction  
- Configured TensorFlow’s **ImageDataGenerator** to handle:
  - Dynamic resizing to `224x224`
  - Normalization using ResNet’s `preprocess_input`
  - Mild augmentation (rotation, shift, zoom, flip)
- Trained using:
  - **Adam optimizer** (learning rate = 2e-4)
  - **Categorical cross entropy loss**
  - **Callbacks** including:
    - EarlyStopping (based on validation loss)
    - ReduceLROnPlateau
    - ModelCheckpoint (saving best weights)

### 3. Fine Tuning and Evaluation
- Unfroze the final convolutional block of ResNet50 for fine tuning.  
- Trained at a lower learning rate (`1e-5`) to improve adaptation to MRI data.  
- Evaluated model performance using:
  - Accuracy and loss curves across epochs  
  - Confusion matrix and classification report for per class metrics  
- Achieved approximately **95% test accuracy** with stable convergence and minimal overfitting.

---

## Key Takeaways
- Transfer learning using **ResNet50** was highly effective for MRI brain tumor classification.  
- Minimal preprocessing, combined with fine tuning, resulted in fast convergence and robust feature learning.  
- The model demonstrated strong generalization across all tumor types with efficient use of computational resources.
---

## Conclusion
This implementation demonstrates the effectiveness of pretrained convolutional networks such as **ResNet50** for medical image classification. The model achieved strong accuracy and interpretability while requiring minimal preprocessing and manual tuning.