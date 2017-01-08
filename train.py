#!/usr/bin/python
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.externals import joblib
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Need to run this using:
#  python train.py ./datafiles/

def cleandata(inputdf, trainfilt, meanvaldf):
	"""
	Parameters 
	___________

	inputdf		:pd.DataFrame:	Input data frame of model variables 

	trainfilt: 	:bool: Boolean for filter to limit mean inputations for training and test data

	meanvaldf: 	:pd.DataFrame: Training data mean values for inputation on test data

	"""
	xdf = inputdf
	# Converting the columns that were read as character to numeric columns
	for col in xdf.columns:
		if xdf[col].dtype not in (float, int) and col!='ad':
			xdf.ix[xdf[col].str.strip()=='?', col] = np.nan
			xdf[col] = xdf[col].astype(float)

	# Replacing the unknown values with missing binary/indicator variables
	for col in xdf.columns:
		if xdf[col].dtype in (float, int) and col!='ad':
			xdf[col+'missing'] = xdf[col].isnull().astype(int)
			if xdf[col].nunique() ==2:
				xdf[col] = xdf[col].fillna(0)
			else:
				if meanvaldf is None:
					meanval = xdf.ix[xdf[col].notnull() & trainfilt, col].mean()
				else:
					meanval = meanvaldf[col]['mean']

				xdf[col] = xdf[col].fillna(meanval)
	return xdf

def main(pth, meanvaldf):
	"""
	Parameters 
	___________

	pth			:string:		Path identifying input data for training
	meanvaldf	:pd.DataFrame: 	Training data mean values for inputation on test data

	"""
	np.random.seed(seed=100)
	inputfile = 'data'
	colfile = 'column.names.txt'

	# Reading in dataset
	dfx = pd.read_csv(os.path.join(pth, inputfile), header=None)
	df2 = pd.read_csv(os.path.join(pth, colfile), sep=":", skiprows=0)
	df2 = df2.reset_index()
	df2.columns = ['variable', 'type']
	dfx.columns = df2['variable'].tolist() + ['ad']
	trainfilt = np.ones(dfx.shape[0]) == 1

	df1 = cleandata(dfx, trainfilt, meanvaldf)

	if meanvaldf == None:
		meanvals = df1.describe()
		meanvals.to_csv('./inputmeanvalues.csv')

	print("Writing summary statistics to ./inputmeanvalues.csv")

	ys = (df1['ad'].str.strip() == 'ad.' ).astype(int).values
	xcols = [col for col in df1.columns if col !='ad']
	X_train, y_train= df1.ix[trainfilt, xcols].values, ys[trainfilt]

	nbmodel = BernoulliNB() 
	nbmodel.fit(X_train, y_train)

	nbclass = nbmodel.predict(X_train)  
	nbproba = nbmodel.predict_proba(X_train)[:, 1]

	# Choosing the model with the best performance on the validation data, which happens to be the Naive Bayes
	print("Train Naive Bayes AUC = %.6f" % roc_auc_score(y_train, nbproba))
	print("Train Naive Bayes Accuracy = %.6f" %  np.mean(y_train==nbclass) )

	# Confusion Matrix
	print("Training Confusion Matrix")
	print(confusion_matrix(y_train, nbclass))

	print("Writing model to ./ad_classifier.pkl" )
	joblib.dump(nbmodel, 'ad_classifier.pkl') 

if __name__ == "__main__":
	path = sys.argv[1]
	main(pth=path, meanvaldf=None)