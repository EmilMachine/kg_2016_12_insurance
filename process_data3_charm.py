import pandas as pd
import pickle
import os.path
import numpy as np

# Settings
base_path = '/Users/emil/Documents/code/kaggle/kg_2016_12_insurance/'
train_set_raw = './train.csv'
train_set_pickled = './train.pickle'

# //////////////////////////////////
### LOAD
### Pickle guard to test things out from smaller pickle
if os.path.isfile(base_path+train_set_pickled):
	### Load picled data
	with open(base_path+train_set_pickled, 'rb') as handle:
		df = pickle.load(handle)
	
else:
	### LOAD FROM CSV and save as picle
	df = pd.read_csv(base_path+train_set_raw)#, nrows=1000)
	### LOAD DATA
	# for BIG DATA SET SEE IF RUN INTO PROBLEMS
	# Remove: # nrows
	# maybe try out:
		# skiprows 
		# chunck sizes 
	with open(base_path+train_set_pickled, 'wb') as handle:
  		pickle.dump(df, handle)


# //////////////////////////////////
### FORMAT
import sklearn.preprocessing as preprocessing
### GET coulmn names and types (from names)
column_names = df.columns.values.tolist()
# get types of columns (LOOKINTO get it from dtype)
categorical_col = [i for i in column_names if 'cat' in i]
contenious_col = [i for i in column_names if 'cont' in i]
target_col = ['loss']
id_col = ['id']
### remove NA
# df = df.dropna()

### VECTORIZE Categorical
# uses get_dummies
# LOOKINTO
# sklearn DictVectoriser
# pandas Categorical
df_cat_dummied = pd.get_dummies(df[categorical_col],sparse=False)#, sparse=True)
###To reindex new observation
#df1 = pd.get_dummies(df[categorical_col], sparse=True)
#df1.reindex(columns = dummies_frame.columns, fill_value=0)


### SCALE Numerical
# LOOKINTO DeprecationWarning X.reshape
#df_num_rescaled = df[contenious_col].apply(lambda x: MinMaxScaler().fit_transform(x))
minmax_scale = preprocessing.MinMaxScaler().fit(df[contenious_col])

df_num_rescaled = df[contenious_col]

df_num_rescaled[contenious_col] = minmax_scale.transform(df[contenious_col])

#print(df_cat_dummied.shape)
#type(df_num_rescaled)
#print(df_num_rescaled.shape)

# merge together again
df_x_train =  pd.concat([df_cat_dummied, df_num_rescaled], axis=1)

## numpy version
#df_x_train =  np.concatenate([np.array(df_cat_dummied), df_num_rescaled], axis=1)

df_y_train = df[target_col]



#train_data = np.array(df[contenious_col])
#train_target = np.array(df[target_col])


## ////////////////////////////////////////
### SIMPLE LINEAR MODEL
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df_x_train,df_y_train)

print(reg.coef_)
# self evaluated
print(reg.score(df_x_train,df_y_train))



### ///////////////////////////////////////
### XGBoost
# requires to train it
import sys
xgboost_path = '/Users/emil/Documents/code/kaggle/xgboost/python-package'
sys.path.append(xgboost_path)
# INSTALL GUIDE http://xgboost.readthedocs.io/en/latest/build.html
import xgboost as xgb

#X_train = df.drop(target_col[0],axis=1)
#X_train = df[contenious_col]
#Y_train = df[target_col]
dtrain = xgb.DMatrix(df_x_train, df_y_train)

## Simple training
#params = {"objective": "reg:linear"}#, "booster":"gblinear"}
#gbm = xgb.train(dtrain=dtrain,params=params)
# todo EVAL

## XGBOOST WITH CROSS VALIDATION
#https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py
from sklearn import cross_validation, metrics

param = {'max_depth':5, 'eta':1, 'silent':1, 'objective':'reg:linear_model', 'n_estimators':'10000'}
num_round = 2
cv_xgb = xgb.cv(param, dtrain, num_round, nfold=5,
       metrics='rmse', seed = 0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])



#metrics.accuracy_score(cv_xgb)
print('done')

### LIST OF PARAMETERS 
#https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

### META PARAMETERS OPTIMIZATION 
#https://www.dataiku.com/learn/guide/code/python/advanced-xgboost-tuning.html



#final_xgb = xgb.train(params, dtrain, num_boost_round = 10)






### Inspect trick to see parameters
#import inspect
#inspect.getargspec(xgb.plot_import)
#
#xgb.plot_importance(final_gb )
#xgb.eval(dtrain)
#print(.head(3))

#########################
### LOAD PREDICT DATA 
# Load data 
in_file = 'test.csv'
out_file = 'submission.csv'

# read test
df_test = pd.read_csv(base_path+in_file) 


# rescale and df_dummy
df_test_dummied = pd.get_dummies(df_test[categorical_col])
a = df_test_dummied.columns.values.flatten().tolist()
b = df_cat_dummied.columns.values.flatten().tolist()
cat_not_seen = set(a).difference(b)
df_test_dummied.drop(cat_not_seen, axis=1, inplace=True)


df_test_dummied2 = df_test_dummied.reindex(columns = df_cat_dummied.columns, fill_value=0)
#a = df_test_dummied.columns.values.flatten().tolist()
#b = df_cat_dummied.columns.values.flatten().tolist()

#print(df_cat_dummied.shape)
#print(df_test_dummied2.shape)


df_test_rescaled = df_test[contenious_col].copy()
df_test_rescaled[contenious_col] = minmax_scale.transform(df_test[contenious_col])

# df_test
# df_test_rescaled.dtypes.names

### REMOVE DUMMY COLS NEVER SEEN IN TRAINING

df_x_test = pd.concat([df_test_dummied2, df_test_rescaled], axis=1)
# numpy way
#df_x_test = np.concatenate([np.array(df_test_dummied), df_test_rescaled], axis=1)



# merge together again
pred_x_test = xgb.DMatrix(df_x_test)
#pred_x_test = xgb.DMatrix(df_x_train)
pred_y_test = final_xgb.predict(pred_x_test)


# get id
pred_id_list = df_test[id_col].values.flatten().tolist()

# format it for writing
final = zip(pred_id_list,list(pred_y_test))
final_string = 'id,loss\n' + '\n'.join(['%i,%f'%i for i in final])

# write
with open(base_path+out_file,'w') as f:
    f.write(final_string)

print('done')

