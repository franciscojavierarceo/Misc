import os
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.externals import joblib
from sklearn.linear_model import LassoCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

pth = './datafiles/'
inputfile = 'data'
colfile = 'column.names.txt'

mymodel = joblib.load('ad_classifier.pkl')

df1 = pd.read_csv(os.path.join(pth, inputfile), header=None)
df2 = pd.read_csv(os.path.join(pth, colfile), sep=":", skiprows=0)
df2 = df2.reset_index()
df2.columns = ['variable', 'type']

df1.columns = df2['variable'].tolist() + ['ad']

for col in df1.columns:
    if df1[col].dtype not in (float, int) and col!='ad':
        df1.ix[df1[col].str.strip()=='?', col] = np.nan
        df1[col] = df1[col].astype(float)

rans = np.random.rand(df1.shape[0]) 
trainfilt = rans <= (1/3.)
validfilt = (rans > (1/3.) ) & (rans <= (2/3.))
testfilt = rans > (2/3.)


for col in df1.columns:
    if df1[col].dtype in (float, int) and col!='ad':
        df1[col+'missing'] = df1[col].isnull().astype(int)
        if df1[col].nunique() ==2:
            df1[col] = df1[col].fillna(0)
        else:
            meanval = df1.ix[df1[col].notnull() & trainfilt, col].mean()
            df1[col] = df1[col].fillna(meanval)


ys = (df1['ad'].str.strip() == 'ad.' ).astype(int).values
xcols = [col for col in df1.columns if col !='ad']
X_train, X_valid, X_test = df1.ix[trainfilt, xcols].values, df1.ix[validfilt, xcols].values, df1.ix[testfilt, xcols].values
y_train, y_valid, y_test = ys[trainfilt], ys[validfilt], ys[testfilt]