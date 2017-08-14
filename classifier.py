from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random


class MyClassifier():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
	
	def predict(self, X_test):
		predictions = []
		for item in X_test:
			target_label = random.choice(y_train)
			predictions.append(target_label)
		return predictions


# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# train tree classifier
clf = MyClassifier()
clf.fit(X_train, y_train)

# print out prediction accuracy
print(accuracy_score(y_test, clf.predict(X_test)))
