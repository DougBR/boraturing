import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras


df = pd.read_csv('onlyvalidCID.csv', sep=',')
df.fillna(0, inplace=True)
for column in df.columns:
    if df[column].dtype == type(object):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

X = df.drop('cid', axis=1)
y = df['cid']
del df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print("Iniciando treinamento")
#print("Y:", y)

#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)

from sklearn import tree
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(20)),
	keras.layers.Dense(5, activation=tf.nn.relu),
	keras.layers.Dense(, activation=tf.nn.softmax)])
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print("Acurácia: ", accuracy_score(y_test, predictions))
