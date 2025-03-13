# Breast Cancer Prediction Using Machine Learning

## Overview

This project aims to build a **Logistic Regression classifier** to predict whether a given case of breast cancer is **malignant** or **benign** based on medical imaging features. The dataset used for training and evaluation is the **Breast Cancer Wisconsin (Diagnostic) Dataset** obtained from Kaggle and the UCI Machine Learning Repository.

## Learning Objectives

By working on this project, you will learn:

- How to build a **Logistic Regression classifier** for breast cancer prediction.
- How to **download datasets directly from Kaggle** using the Kaggle API.
- How to set up and work with **Google Colab** for training and evaluating machine learning models.

---

## Project Workflow

### **Task 1: Introduction and Import Libraries**

- Understanding the project goal and scope.
- Importing necessary Python libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

### **Task 2: Download Dataset from Kaggle**

- Using the **Kaggle API** to directly fetch the dataset into Google Colab.
- Extracting and loading the dataset.

### **Task 3: Load & Explore the Dataset**

- Loading the dataset into a Pandas DataFrame.
- Checking for missing values, duplicates, and outliers.
- Performing exploratory data analysis (EDA) with visualizations.

### **Task 4: Perform Label Encoding**

- Encoding the **Diagnosis column** (M = 1, B = 0) to prepare data for machine learning algorithms.

### **Task 5: Data Preprocessing**

- Splitting the dataset into **independent features (X)** and **dependent variable (Y)**.
- Performing **Feature Scaling** to normalize numerical values.
- Splitting data into **training and testing sets**.

### **Task 6: Build a Logistic Regression Classifier**

- Training a Logistic Regression model using Scikit-learn.
- Fine-tuning model hyperparameters.

### **Task 7: Model Evaluation**

- Evaluating the performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

---

## **Dataset Information**

### **Source**

- **Kaggle Dataset:** [Breast Cancer Wisconsin (Diagnostic) Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- **UCI Machine Learning Repository:** [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

### **Description**

The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. These features describe the characteristics of the cell nuclei present in the image.

The dataset was first introduced in:

- **K. P. Bennett and O. L. Mangasarian**: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34.

### **Attribute Information**

The dataset consists of **569 samples**, each with **32 attributes**:

- **ID number**
- **Diagnosis (M = malignant, B = benign)**
- **30 real-valued features** computed for each cell nucleus:
  1. **Radius** (mean distance from center to perimeter)
  2. **Texture** (standard deviation of gray-scale values)
  3. **Perimeter**
  4. **Area**
  5. **Smoothness** (local variation in radius lengths)
  6. **Compactness** ((perimeter^2 / area) - 1.0)
  7. **Concavity** (severity of concave portions of the contour)
  8. **Concave Points** (number of concave portions of the contour)
  9. **Symmetry**
  10. **Fractal Dimension** (coastline approximation - 1)

Each feature has three computed statistics:

- **Mean**
- **Standard error**
- **Worst (mean of the three largest values)**

### **Missing Values**

- There are **no missing values** in the dataset.

### **Class Distribution**

- **357 benign cases** (label = 0)
- **212 malignant cases** (label = 1)

---

## **Tools & Libraries Used**

- **Python 3.7+**
- **Google Colab / Jupyter Notebook**
- **NumPy** (Data manipulation)
- **Pandas** (Data preprocessing & analysis)
- **Matplotlib & Seaborn** (Data visualization)
- **Scikit-learn** (Machine learning & evaluation)
- **Kaggle API** (Dataset retrieval)

---

## **Installation & Setup**

### **Step 1: Clone Repository & Install Dependencies**

```bash
# Clone the repository
git clone https://github.com/patelmj04/breast-cancer-prediction.git
cd breast-cancer-prediction

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Download Dataset via Kaggle API**

```bash
# Install Kaggle API
pip install kaggle

# Authenticate Kaggle (upload kaggle.json to your environment)
kaggle datasets download -d uciml/breast-cancer-wisconsin-data

# Unzip dataset
unzip breast-cancer-wisconsin-data.zip
```

### **Step 3: Run the Notebook**

- Open **Google Colab** or Jupyter Notebook.
- Upload the dataset and run the notebook file.

---

## **Results & Findings**

- The **Logistic Regression classifier** achieves high accuracy in classifying breast cancer cases.
- The **evaluation metrics** show that the model performs well in distinguishing between malignant and benign cases.
- Feature selection and hyperparameter tuning can further improve performance.

---

## **Future Enhancements**

- Implement other machine learning models like **Random Forest, SVM, and Deep Learning**.
- Perform **feature selection** to improve efficiency.
- Deploy the model using **Flask / FastAPI** as a web application.
- Enhance dataset visualization for better interpretability.

---

## **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

## **Acknowledgments**

- **Kaggle** for providing the dataset.
- **UCI Machine Learning Repository** for dataset insights.
- **Scikit-learn & Pandas community** for excellent ML tools.
