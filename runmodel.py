#!/usr/bin/python
import os
import sys
import train
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib

def main(traindatapth, testdatapath, inputfile):
	"""
	Parameters 
	___________
	traindatapth	:string:		Path identifying input data for training
	testdatapath	:string:		Path identifying input data for model execution
	inputfile		:string:		Path identifying input data file for model execution

	"""

	if 'ad_classifier.pkl' not in os.listdir(testdatapath):
		print("Training model")
		train.main(traindatapth, meanvaldf=None)
	else:
		print("Loading pretrained model located %s" % os.path.join(testdatapath, 'ad_classifier.pkl') )

	meanvaldf = pd.read_csv(os.path.join(testdatapath, 'inputmeanvalues.csv'), index_col=0)
	colfile = 'column.names.txt'

	# Reading in dataset
	dfx = pd.read_csv(os.path.join(traindatapth, inputfile), header=None)
	df2 = pd.read_csv(os.path.join(traindatapth, colfile), sep=":", skiprows=0)
	df2 = df2.reset_index()
	df2.columns = ['variable', 'type']

	if dfx.shape[1] == (df2.shape[0] + 1):
		dfx.columns = df2['variable'].tolist() + ['ad']
	
	if dfx.shape[1] != (df2.shape[0] + 1):
		assert dfx.shape[1] != df2.shape[0], "Input data has wrong number of columns"

	df1 = train.cleandata(dfx, None, meanvaldf)

	nbmodel = joblib.load('ad_classifier.pkl')

	xcols = [col for col in df1.columns if col !='ad']
	X_train= df1[xcols].values

	# Exporting predctions
	preds = pd.DataFrame(nbmodel.predict(X_train)  , columns=['preds'])
	preds.to_csv("./preds.csv")
	print("Predictions exported to ./preds.csv")

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3])