# SeekTruth: Naive Bayes Classifier for Review Authenticity

## Overview

This project, titled "SeekTruth," involves developing a Naive Bayes classifier to distinguish between deceptive and truthful hotel reviews. The classifier uses a bag-of-words model to analyze text data and predict review authenticity.

## Data Processing

* Data Loading: The system loads the dataset from text files where each line represents a review followed by its label (truthful or deceptive).
* Tokenization: The text of each review is tokenized into individual words. This process involves converting the text to lowercase to standardize it and possibly removing punctuation.

  
## Feature Engineering
* Bag of Words Model: The system constructs a bag of words model which involves counting the occurrences of each word in each review. This model ignores the order of words but keeps track of their frequency.
* Stop Words Removal: Common words (stop words) that are unlikely to contribute to the authenticity analysis are removed. This step improves the focus on more meaningful words.
* Feature Selection: Some methods of reducing the dimensionality of the feature set may be employed, such as removing very rare words that appear in very few documents.
  
## Naive Bayes Classifier
* Probability Estimation: For each class (truthful or deceptive), the system calculates the prior probability of the class and the conditional probabilities of each word given the class.
* Training: Using the training data, the classifier learns the probabilities associated with each word in the context of truthful and deceptive reviews.
* Smoothing: To handle the problem of zero probability in case a word in the test set has not been seen in the training set, smoothing techniques like Laplace smoothing are used.
  
## Testing and Classification
* Logarithmic Probabilities: To avoid underflow problems common with small probabilities, the system uses logarithms of probabilities. This approach transforms products into sums, simplifying the calculations.
* Decision Rule: For each review in the test dataset, the classifier calculates the total log probability for each class and classifies the review based on the highest probability.
  
## Performance Evaluation
* Accuracy Measurement: After classifying the test data, the system evaluates its performance by comparing the predicted labels with the true labels provided in the test dataset.
* Error Analysis: Possible steps to analyze errors and misclassifications to understand the limitations of the model.

