import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

money_data_frame = pd.read_csv('money.csv')

happines_data_frame = pd.read_csv('happiness.csv')

result = pd.merge(money_data_frame, happines_data_frame, how="inner", on="Country")

result['GDP'] = result['GDP'].apply(float)
print(result.head(10))
# Plot the chart
# plt.style.use('classic')
# result.plot(kind='scatter', x='GDP', y='Happiness')
# plt.show()

x = np.array(result['GDP']).reshape(-1, 1)
y = result['Happiness']
should = np.array([8000.0, 70000.0]).reshape(-1, 1)
print(should)
clf = LinearRegression().fit(x, y)

print(clf.predict(should))
print(clf.score(x, y))