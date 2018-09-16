from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

clf = joblib.load('trainedtree80train.dat')
X = pd.read_csv('testData80_allfeatures1_10000.csv', sep=',')
#X.drop('base_hackaturing.cid', axis=1)
X.fillna(0, inplace=True)
for column in X.columns:
    if X[column].dtype == type(object):
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))

new_cids = clf.predict(X)

pd.concat([X, new_cids], axis=1)

X.to_csv('cidsfound1_10000', sep=',')
#df1['e'] = Series(np.random.randn(sLength), index=df1.index)
