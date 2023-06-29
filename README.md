# Product Feedback Classifier


This repository contains a machine learning project that focuses on classifying product feedback into different categories. The project utilizes natural language processing techniques and the Naive Bayes algorithm to predict the category of feedback based on its content.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Overview
Product feedback plays a crucial role in understanding customer sentiment and improving products and services. This project aims to automate the categorization of feedback to efficiently process and analyze large volumes of customer comments.

The classifier is implemented using Python, making use of the NLTK library for natural language processing tasks. The data is preprocessed by tokenizing, lowercasing, removing stopwords, and lemmatizing. The preprocessed text is then transformed into numerical feature vectors using the CountVectorizer from scikit-learn. The Naive Bayes algorithm is employed for training the classifier, which is capable of predicting the category of new, unseen feedback.

## Dataset
The project currently utilizes a predefined dataset included within the code file. It consists of sample product feedback comments and their associated labels. However, to achieve optimal results, it is recommended to replace the provided dataset with your own domain-specific feedback data.

## Preprocessing
The script applies several preprocessing steps to clean and normalize the text data before training the classifier:

1. Tokenization: The comments are split into individual words using the NLTK `word_tokenize()` function.
2. Lowercasing: All words are converted to lowercase using a list comprehension.
3. Stopword Removal: Common English stopwords are removed from the list of words using the NLTK `stopwords` set.
4. Lemmatization: Each word is lemmatized using the NLTK `WordNetLemmatizer` to convert them to their base form.

You can modify or extend these preprocessing steps to suit your specific requirements.

## Training and Evaluation
The project employs the Naive Bayes algorithm, implemented through the `MultinomialNB` class from scikit-learn, for training the classifier. The training data is split into a training set and a testing set using scikit-learn's `train_test_split` function. The model is trained on the training set and evaluated on the testing set using the `classification_report` function from scikit-learn's `metrics` module.

## Results
The classification report provides performance metrics such as precision, recall, F1-score, and support for each category. These metrics offer insights into the classifier's accuracy and effectiveness in categorizing the product feedback.
