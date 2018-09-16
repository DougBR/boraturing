import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


df = pd.read_csv('onlyvalidCID.csv', sep=',')
df.fillna(0, inplace=True)
for column in df.columns:
    if df[column].dtype == type(object):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

X = df.drop('cid', axis=1)
y = df['cid']
del df
file = open('traintestPoints.py', 'w')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)
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
print("Fitting completed!")

# Salvando máquina treinada
joblib.dump(my_classifier, 'trainedtree20train.dat')
print("Model saved successfully")
try:
	predictions = my_classifier.predict(X_test)
	predictions("predictions realized with success!")
except Exception as e:
	logger.error('Error in the pedicting fase: '+ str(e))
finally:
	print("Past prediction line")

from sklearn.metrics import accuracy_score
print("Accuracy Decision Tree (20% treinamento): ", accuracy_score(y_test, predictions))

del predictions

 
del my_classifier

##print("Treinando random forest")
##clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
##clf = clf.fit(X_train, y_train)
##print("Acurácia Random Forest:", clf.score(X_test, y_test))
##filename = 'trainedRNDTree.dat'
##pickle.dump(clf, open(filename, 'wb'))
## 
### load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)



