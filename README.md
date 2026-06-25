# Fake_News_Classifier_LSTM
Fake News Detection using TF-IDF, Random Forest, LSTM, and BiLSTM with performance comparison and hyperparameter tuning.

## Overview

This project focuses on detecting fake news articles using both traditional Machine Learning techniques and Deep Learning models. The objective is to compare the performance of feature-based approaches such as Count Vectorization and TF-IDF with neural network architectures like LSTM and Bidirectional LSTM.

## Dataset

Dataset: WELFake_Datset

The dataset contains news articles labeled as:

- Real News (1)
- Fake News (0)

Each record consists of:

- News title
- News content
- Label

## Exploratory Data Analysis (EDA)

The following analyses were performed:

- Class distribution visualization
- News length distribution
- Missing value detection
- Duplicate record analysis

## Text Preprocessing

The following preprocessing steps were applied:

- Lowercase conversion
- HTML tag removal
- URL removal
- Punctuation removal
- Stopword removal
- Tokenization
- Text normalization

## Feature Engineering
### Count Vectorization

- News articles were converted into numerical representations using CountVectorizer.

### TF-IDF Vectorization

- TF-IDF was used to capture the importance of words across documents.

## Machine Learning Models

The following models were implemented:

### Random Forest Classifier
- Count Vectorizer + Random Forest
- TF-IDF + Random Forest
- TF-IDF + Logistic Regression
  
### Hyperparameter Tuning
GridSearchCV was used to optimize:

- Number of estimators
- Maximum tree depth
- Minimum samples split

## Deep Learning Models
### LSTM

Architecture:

- Embedding Layer → LSTM → Fully Connected Layer → Output
- Embedding Layer → LSTM → Dropout (0.3) → Fully Connected Layer → Output

### Bidirectional LSTM

Architecture:

Embedding Layer → Bidirectional LSTM → Dropout (0.3) → Fully Connected Layer → Output

Additional techniques:

- Padding
- Vocabulary construction
- Word-to-index conversion
- PyTorch Dataset and DataLoader
- Dropout
- Adam Optimizer
- Binary Cross Entropy Loss

## Results

| Model | Accuracy |
|--------|----------|
| Count Vectorizer + Random Forest | 88.78% |
| TF-IDF + Random Forest | 90.64% |
| Tuned Random Forest | 93.47% |
| TF-IDF + Logistic Regression | 95.68% |
| LSTM | 96.98% |
| BiLSTM | 96.89% |

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- PyTorch
- Matplotlib
- NLTK

## Conclusion

The project compares traditional machine learning methods with deep learning approaches for fake news detection. Results demonstrate the effectiveness of text preprocessing, feature engineering, and sequence modeling techniques in identifying misleading news articles.
