# Importing the required packages for Sentiment Analysis
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
# % matplotlib inline

# Load the dataset and read the top 5 dataset using pandas head function

df = pd.read_csv('Reviews.csv')
df.head()

print(df.describe())

# We are going to drop some columns which we not required for analysis.
data = df.drop(labels=["ProductId", "ProfileName","HelpfulnessNumerator","HelpfulnessDenominator","Time"], axis=1)
print(data)

# Data Explanation of Score
sns.countplot(data['Score']) 
plt.show()

# We create a new column postive_negative which describe about the positivity (1) or negativity (0) of the reviews.
data.dropna(inplace=True)
data[data['Score'] != 3]
data['postive_negative'] = np.where(data['Score'] > 3, 1, 0)
print(data.head())

# Creating wordclouds for positivity and negativity to know which words has more values in the reviews
# For that we create two different dataframes (positivity and negativity)
positive_reviews = data[data['postive_negative'] == 1]
negative_reviews = data[data['postive_negative'] == 0]

# We have seperated all positive reviews dataset from the dataframe data to positive_reviews
print(positive_reviews)

# We have seperated all negative reviews dataset from the dataframe data to negative_reviews
print(negative_reviews)

# Wordcloud — positive_reviews

# Create new stopword list
stopwords = set(STOPWORDS)
pos = " ".join(review for review in positive_reviews.Summary)
# pos
wordcloud = WordCloud(stopwords=stopwords).generate(pos)
# # Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Wordcloud — negative_reviews
pos = " ".join(review for review in negative_reviews.Summary)# pos
wordcloud = WordCloud(stopwords=stopwords).generate(pos)
# # Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Printing the dataframe
print(data)

# Building the Model
X_train, X_test, y_train, y_test = train_test_split(data['Summary'], data['postive_negative'], random_state = 0)

# As now we have binary dependent varibale and we can apply the logistic regression algorithm on that but
# the logistic regression will not understand the text, so we need to convert the text to matrix

vectorizer = CountVectorizer()
vector = vectorizer.fit(X_train)
print(vector)

print(len(vector.get_feature_names()))

# We going to create matrix of the text
X_train_Matrix = vector.transform(X_train)
print(X_train_Matrix)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train_Matrix, y_train)
# Prediction
prediction = model.predict(vector.transform(X_test))
# Testing
print(classification_report(prediction,y_test))

# Testing
print(model.predict(vector.transform(['The food is tasty', 'The food is not bad, I will buy them again'])))

"""
Now, we are going to re-train the model. But before that, we are going to do some pre-processing on the text.
"""

# We are now going to remove the those words who comes less than 6 times in a document
vector = TfidfVectorizer(min_df = 6).fit(X_train)
print(len(vector.get_feature_names()))

X_train_TF = vector.transform(X_train)
model = LogisticRegression()
model.fit(X_train_TF, y_train)

# Prediction
prediction = model.predict(vector.transform(X_test))

# Testing
print(classification_report(prediction,y_test))

# Testing
print(model.predict(vector.transform(['The food is tasty', 'The food is not bad, I will buy them again'])))

