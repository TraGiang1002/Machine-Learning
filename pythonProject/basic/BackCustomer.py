
import pandas as pd
import numpy as np
from pip._internal.utils.misc import tabulate

data = pd.read_csv(filepath_or_buffer="Bank Customer Churn Prediction.csv", sep=",", header=0).head(100)
new_df = tabulate(data, headers='keys', tablefmt='psql')
print(new_df)