import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

comments = [
    "The product arrived late and the packaging was damaged.",
    "The instructions provided were unclear and hard to follow.",
    "I received the wrong item in the package.",
    "The product quality is poor, it broke after a few uses.",
    "The customer service was unresponsive and did not resolve my issue.",
    "The shipping cost was too high compared to other sellers.",
    "The product did not match the description on the website.",
    "The product was missing some parts.",
    "The product is overpriced for its quality.",
    "The packaging was not sturdy enough, and the product got damaged."
]

labels = [
    "Shipping",
    "Instructions",
    "Shipping",
    "Product Quality",
    "Customer Service",
    "Shipping",
    "Product Description",
    "Product",
    "Price",
    "Packaging"
]

# Combine comments and labels into a list of tuples
data = list(zip(comments, labels))

# Shuffle the data randomly
# This randomization helps in avoiding any order-related biases in the data.
random.shuffle(data)

# Print the sample data
for comment, label in data:
    print(f"Comment: {comment}")
    print(f"Label: {label}")
    print()


"""
This function takes a comment as input 
and performs several preprocessing steps 
on it to clean and normalize the text.
"""


def preprocess_comment(comment):
    # Tokenization: The comment is split into individual words using the word_tokenize() function.
    tokens = word_tokenize(comment)

    # Lowercasing: All words are converted to lowercase using a list comprehension.
    lowercase_tokens = [token.lower() for token in tokens]

    # Stopword removal: Common English stopwords are removed from the list of words using the stopwords set.
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lowercase_tokens if token not in stop_words]

    # Lemmatization: Each word is lemmatized using the WordNetLemmatizer to convert them to their base form.
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the preprocessed tokens back into a single string
    preprocessed_comment = ' '.join(lemmatized_tokens)

    return preprocessed_comment


# Example usage
for comment, label in data:
    preprocessed_comment = preprocess_comment(comment)
    print(f"Comment: {preprocessed_comment}")
    print(f"Label: {label}")
    print()

# List of preprocessed comments
preprocessed_comments = [preprocess_comment(comment) for comment, _ in data]

# Initialize CountVectorizer
# This class is responsible for converting text data into numerical feature vectors.
vectorizer = CountVectorizer()

# Fit and transform the preprocessed comments
features = vectorizer.fit_transform(preprocessed_comments)

# Convert the feature matrix to an array
feature_array = features.toarray()

# Print the feature matrix
print(feature_array)

# List of preprocessed comments and labels
preprocessed_comments = [preprocess_comment(comment) for comment, _ in data]
labels = [label for _, label in data]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the preprocessed comments
features = vectorizer.fit_transform(preprocessed_comments)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = classifier.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))
