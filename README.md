# Hate-Speech Classification using Baseline Models and Bidirectional LSTM

## Objective

This project focuses on Hate Speech Detection and Classification utilizing **`Bidirectional LSTM.`** The comprehensive approach includes dataset exploration, preprocessing, baseline models like **`Decision Tree, Random Forest, AdaBoost,`** and a **`Bidirectional LSTM`** model, achieving approximately **`79-80% accuracy.`** The goal is to contribute to mitigating online hate speech.

## Dataset

- **Entries**: **`24,783`**
- **Columns**: **`'count', 'hate_speech', 'offensive_language', 'neither', 'class', and 'tweet'`**
- **Class distribution**: **`0 (1,430 instances) 5.76% , 1 (19,190 instances) 77.45%, 2 (4,163 instances) 16.79%`**
- **Data types**: Numeric and text
- **`No null values`** in the dataset; 'tweet' text converted to lowercase.

## Model Architecture

### Dataset Exploration and Preprocessing

#### Overview:

- **Rows**: **`24,783`**
- **Columns**: **`7 (int64: 6, object: 1)`**
- **Dtype**: **`int64 (6), object (1)`**
- **Memory Usage**: 1.3+ MB

#### Data Info:

- **`Non-null`** entries in all columns

#### Descriptive Stats:

- **`Mean hate speech: 0.28, offensive language: 2.41, neither: 0.55`**

#### Selected Data:

- 'class' and 'tweet' columns retained

#### Class Distribution:

- **`Labels: 0 (1,430), 1 (19,190), 2 (4,163)`**

#### Null Values:

- **`No null values`**

#### Text Preprocessing:

- Lowercased `tweet` column
- Removed punctuation
- Additional cleaning using a custom function

#### Stopword Removal:

- Removed `stopwords` from the `tweet` column using NLTK stopwords

#### Bag of Words Approach:

- Utilized **`CountVectorizer`** with **`max_features=75`** for final data preparation
- Converted `tweet` text into numerical features

### Data Preparation

- Employed **`stratified`** split **`(test_size=0.3)`** for training and testing datasets

### Baseline ML Models

#### Decision Tree Classifier:

- Accuracy: **`0.79`**

#### Random Forest Classifier:

- Utilized **`10 estimators`**, resulting in an accuracy of **`0.83`**

#### AdaBoost Classifier:

- Employed **`100 estimators`**, achieving an accuracy of **`0.84`**

### Deep Neural Network - Bidirectional LSTM Model Training Overview

#### Model Architecture:

- **Embedding Layer**:
  - Input Length: Tailored to match the shape of X_train (input text sequences)

- **SpatialDropout1D Layer**:
  - Introduces dropout in the embedding layer to enhance model robustness
  - **`Dropout Rate: 0.2`**

- **Bidirectional LSTM Layer**:
  - Employs a **`Bidirectional Long Short-Term Memory (LSTM)`** layer for capturing sequence information bi-directionally
  - **`Units: 20`**
  - **`Dropout: 0.2`**
  - **`Recurrent Dropout: 0.2`**

- **Dense Layer**:
  - Utilizes a dense layer for final classification
  - Units: **`3`** (matching the number of classes)
  - Activation Function: **`softmax`**

#### Compilation:

- **Loss Function**:
  - **`Categorical Crossentropy`**: Suitable for multi-class classification tasks

- **Optimizer**:
  - **`Adam Optimize`**r: Adaptive optimization algorithm for efficient training

- **Metrics**:
  - **`Accuracy`**: Monitored metric for model performance

#### Training Parameters:

- **Epochs**:
  - **`25`**: Iterations through the entire training dataset

- **Batch Size**:
  - **`64`**: Number of samples processed before updating the model

