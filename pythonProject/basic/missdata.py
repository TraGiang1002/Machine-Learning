import numpy as np
import pandas as pd
import sklearn
print(sklearn.__version__)

from sklearn.preprocessing import SimpleImputer

data = pd.read_csv('missdata.csv', header=None)
print(data)
x = data.values
imp = SimpleImputer(missing_values= np.nan, strategy='mean')
# median || most_frequent
imp.fit(x)
result = imp.transform(x)
print(result)