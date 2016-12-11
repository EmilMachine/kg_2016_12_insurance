import pandas as pd
import pickle
import os.path
import numpy as np

base_path = '/Users/emil/Documents/code/kaggle/kg_2016_12_insurance/'
train_set_raw = './train.csv'
train_set_pickled = './train.pickle'

### Pickle guard to test things out from smaller pickle
if os.path.isfile(base_path+train_set_pickled):
	### Load picled data
	with open(base_path+train_set_pickled, 'rb') as handle:
		df = pickle.load(handle)
	
else:
	### LOAD FROM CSV and save as picle
	df = pd.read_csv(base_path+train_set_raw, nrows=100)
	### LOAD DATA
	# for BIG DATA SET SEE IF RUN INTO PROBLEMS
	# Remove: # nrows
	# maybe try out:
		# skiprows 
		# chunck sizes 
	with open(base_path+train_set_pickled, 'wb') as handle:
  		pickle.dump(df, handle)

### GET coulmn names and types (from names)
column_names = df.columns.values.tolist()
# get types of columns
categorical_col = [i for i in column_names if 'cat' in i]
contenious_col = [i for i in column_names if 'cont' in i]
target_col = ['loss']


train_data = np.array(df[contenious_col])
train_target = np.array(df[target_col])

#print(train_data[:2,:])

## ////////////////////////////////////////
### SIMPLE LINEAR MODEL
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(train_data,train_target)

print(reg.coef_)
# self evaluated
print(reg.score(train_data,train_target))



### ///////////////////////////////////////
### XGBoost
# requires to train it
import sys
xgboost_path = '/Users/emil/Documents/code/kaggle/xgboost/python-package'
sys.path.append(xgboost_path)
# INSTALL GUIDE http://xgboost.readthedocs.io/en/latest/build.html
import xgboost as xgb

#X_train = df.drop(target_col[0],axis=1)
X_train = df[contenious_col]
Y_train = df[target_col]
dtrain = xgb.DMatrix(X_train, Y_train)

## Simple training
params = {"objective": "reg:linear"}#, "booster":"gblinear"}
gbm = xgb.train(dtrain=dtrain,params=params)
# todo EVAL

## XGBOOST WITH CROSS VALIDATION
#https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear'}
num_round = 2
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed = 0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])


#print(.head(3))


### TODO categorical encoding


### TODO first do basic traning on numeric variables

