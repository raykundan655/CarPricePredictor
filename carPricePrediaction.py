import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error
import pickle

file=pd.read_csv("C:\\Users\\USER\\Downloads\\4th sem\\py\\CarPrice_Assignment.csv")

# print(file.isnull().sum())

# print(file.dtypes)

print(file.columns)

# sns.heatmap(file.corr(numeric_only=True),annot=True)
# plt.show()

x=file[['CarName','fueltype','carbody','enginesize','stroke','horsepower']]
y=file[['price']]

# why random forest beacuse it can hendal categorical data 

# Random Forest in sklearn does not natively handle string-type categorical variables.
# You must encode them first:
# from sklearn.preprocessing import LabelEncoder
# It’s used to convert categorical text data into numeric labels, 
#  LabelEncoder()->Creates an instance of the LabelEncoder class from sklearn.preprocessing-> .fit_transform(x['CarName']) this function will assign unique value to category and transform
# fit(): Learns all the unique values in the CarName column and assigns each one a unique number.
# transform(): Replaces each car name in the column with its corresponding number.
# eg:
# fit_transform sort cut
# fit() → Understand or learn from the data
# Example: Finds unique categories like ['red', 'green', 'blue']

# transform() → Convert those categories into numbers
# Example: ['red', 'green', 'blue'] → [2, 1, 0]



carname_le = LabelEncoder()
fueltype_le = LabelEncoder()
carbody_le = LabelEncoder()
# This code creates separate objects (instances) of LabelEncoder — one for each categorical column.

x['CarName'] = carname_le.fit_transform(x['CarName'])
x['fueltype'] = fueltype_le.fit_transform(x['fueltype'])
x['carbody'] = carbody_le.fit_transform(x['carbody'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestRegressor()
model.fit(x_train,y_train)

# print(model.predict(['audi','gas','convertible',130,2.68,154]))
# can't direclt insert raw string we have to transform


# input_data =[[
#     carname_le.transform(['audi fox'])[0],   #This simply extracts the first value from that array.
#     #  .transform() always returns an array, even for a single input.
#     fueltype_le.transform(['gas'])[0],
#    carbody_le.transform(['convertible'])[0],
#     130,
#     2.68,
#     154
# ]]
# "Jis object ke through encode kar rahe hain, usi object ke through custom input ko bhi encode karna hoga."
# .transform()	Applies the already-learned mapping	On test data or new input

# print(model.predict(input_data))

y_pred=model.predict(x_test)
print("R2_Score",r2_score(y_test,y_pred))

# save the model and encoder as pickle file

with open("model.pkl","wb") as f:
    pickle.dump(model,f)

# open("model.pkl", "wb")
# This opens (or creates) a file named model.pkl
# The "wb" mode means:
# w → write mode (you are writing to the file)
# b → binary mode (required for pickle, because it saves in binary format, not plain text)
# as f:
# This gives the open file a nickname → f
# f is now the file object — you will use this to write into the file

# pickle.dump(...) means:
# → Save (or “serialize”) a Python object into a file
# model is the object you want to save (in this case, your trained model)
# f is the file you’re saving into — it's a shortcut name for model.pkl while it's open

with open("carname_encoder.pkl","wb") as f:
     pickle.dump(carname_le, f)

with open("fueltype_encoder.pkl","wb") as f:
     pickle.dump(fueltype_le, f)

with open("carbody_encoder.pkl", "wb") as f:
    pickle.dump(carbody_le, f)
















