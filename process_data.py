import pandas as pd
import pickle
import os.path
import numpy as np

train_set_raw = './train.csv'
train_set_pickled = './train.pickle'

### Pickle guard to test things out from smaller pickle
if os.path.isfile(train_set_pickled):
	### Load picled data
	with open(train_set_pickled, 'rb') as handle:
		df = pickle.load(handle)
	
else:
	### LOAD FROM CSV and save as picle
	df = pd.read_csv(train_set_raw, nrows=100)
	### LOAD DATA
	# for BIG DATA SET SEE IF RUN INTO PROBLEMS
	# Remove: # nrows
	# maybe try out:
		# skiprows 
		# chunck sizes 
	with open(train_set_pickled, 'wb') as handle:
  		pickle.dump(df, handle)

### GET coulmn names and types (from names)
column_names = df.columns.values.tolist()
# get types of columns
categorical_col = [i for i in column_names if 'cat' in i]
contenious_col = [i for i in column_names if 'cont' in i]
target_col = ['loss']




train_numeric = np.array(df[contenious_col + target_col])
print(train_numeric[:2,:])


#print(.head(3))


### TODO categorical encoding


### TODO first do basic traning on numeric variables


