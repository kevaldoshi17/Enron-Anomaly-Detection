import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

with open ("finance_dataset.pkl", "rb") as data_file:
	data = pickle.load(data_file)

df = pd.DataFrame(data)
df = df.transpose()
# df.to_csv("processed.csv")
# print(df.columns)

# df1 = pd.read_csv('processed.csv', header = 0)
df2 = df.drop(columns = ['email_address', 'poi'])
# print(df2)
X = df2.to_numpy()
# print(X.shape)
Y = []
for i in df['poi']:
	if str(i) == 'False':
		Y.append(0)
	else:
		Y.append(1)
# print(Y)
Y = np.array(Y)
Y_data = np.reshape((np.asarray(Y)), (len(Y), 1))
# print(Y_data.shape)
 
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_data = imputer.fit_transform(X)

# print(X_data)
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=42)

def NB(X_train, y_train, X_test, y_test):
	gnb = GaussianNB()
	gnb.fit(X_train, y_train)
	predictions = gnb.predict(X_test)
	accuracy  = gnb.score(X_test, y_test)
	clr = classification_report(y_test, predictions)
	print("Naive Bayes Accuracy Score\n",accuracy)
	print("Naive Bayes Classification Report\n", clr)

def LR(X_train, y_train, X_test, y_test):
	clf = LogisticRegression(random_state=0).fit(X_train, y_train)
	predictions = clf.predict(X_test)
	accuracy  = clf.score(X_test, y_test)
	clr = classification_report(y_test, predictions)
	print("Logistic Regression Accuracy score\n",accuracy)
	print("Logistic Regression Classification Report\n", clr)

def SVM(X_train, y_train, X_test, y_test):
	clf = svm.SVC()
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	accuracy  = clf.score(X_test, y_test)
	clr = classification_report(y_test, predictions)
	print("SVM Accuracy Score\n",accuracy)
	print("SVM Classification Report\n", clr)



NB(X_train, y_train, X_test, y_test)
LR(X_train, y_train, X_test, y_test)
SVM(X_train, y_train, X_test, y_test)

