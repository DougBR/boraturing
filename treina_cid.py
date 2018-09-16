import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('onlyvalidCID.csv', sep=',')
df.fillna(0, inplace=True)
for column in df.columns:
    if df[column].dtype == type(object):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

X = df.drop('cid', axis=1)
y = df['cid']
del df
file = open('Failed.py', 'w')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print("Iniciando treinamento")
file.write("X_train:\n"+ str(X_train))
file.write("\nY_train:\n"+ str(y_train))
file.write("\nX_test:\n"+ str(X_test))
file.write("\nY_test:\n"+ str(y_test))


#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print("Acurácia Decision Tree: ", accuracy_score(y_test, predictions))


# Salvando máquina treinada
filename = 'trainedtree.dat'
pickle.dump(my_classifier, open(filename, 'wb'))
 

print("Treinando random forest")
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf = clf.fit(X_train, y_train)
print("Acurácia Random Forest:", clf.score(X_test, y_test))
filename = 'trainedRNDTree.dat'
pickle.dump(clf, open(filename, 'wb'))
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)



