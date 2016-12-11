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


df2 = df.dropna() #= npf_raw[np.isfinite(npf_raw).all(axis=1)]
print(df.shape)
print(df2.shape)
df_with_dummies = pd.get_dummies(df2[categorical_col])

#print(df.isnull().any())

from sklearn import linear_model

## ////////////////////////////////////////
### SIMPLE LINEAR MODEL
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(train_data,train_target)

print(reg.coef_)
# self evaluated
print(reg.score(train_data,train_target))

from sklearn.preprocessing import MinMaxScaler 

df_num_rescaled = df2[contenious_col].apply(lambda x: MinMaxScaler().fit_transform(x))

df_x_train =  pd.concat([df_with_dummies, df_num_rescaled], axis=1)

print(df_x_train.max(axis=1))

df_y_train = df2[target_col]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(df_x_train, df_y_train)

# should be test / validation set
print("Mean squared error: %.2f"
      % np.mean((regr.predict(df_x_train) - df_y_train) ** 2))


###
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
###

###
# http://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-of-certain-column-is-nan
###


#print(.head(3))


### TODO categorical encoding


### TODO first do basic traning on numeric variables

