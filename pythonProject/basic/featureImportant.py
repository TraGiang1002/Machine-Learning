from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as pit

wine = load_wine()
x = wine["data"]
y = wine["target"]
names = wine["target_names"]
rf = RandomForestRegressor()
rf.fit(x, y)
forest_importances = pd.Series(rf.feature_importances_, index=wine.feature_names)
print(forest_importances)
std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis=0)
print(std)