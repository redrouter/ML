
import pandas as pd

msg = pd.read_csv('D:/STUDY/MSC IT - II/Practical All Subject/ML/prac9/naivetext.csv', names=['message', 'label'])
print('The dimensions of the dataset', msg.shape)

msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum

print(X)
print(y)

# Splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

print(xtest.shape)
print(xtrain.shape)
print(ytest.shape)
print(ytrain.shape)

# Output of count vectorizer is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

# Print feature names
print(count_vect.get_feature_names())

# Convert sparse matrix to DataFrame for better representation
df_train = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names_out())
print(df_train)  # Tabular representation
print(xtrain_dtm)  # Sparse matrix representation

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

# Printing accuracy metrics
from sklearn import metrics
print('Accuracy metrics')
print('Accuracy of the classifier is', metrics.accuracy_score(ytest, predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, predicted))
print('Recall and Precision ')
print(metrics.recall_score(ytest, predicted))
print(metrics.precision_score(ytest, predicted))

# Example predictions for new data
docs_new = ['I like this place', 'My boss is not my savior']
X_new_counts = count_vect.transform(docs_new)
predicted_new = clf.predict(X_new_counts)

for doc, category in zip(docs_new, predicted_new):
    print('%s -> %s' % (doc, category))




