import pandas as pd
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image, display
import pydotplus
import os

os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'

random_forest_dt = pd.read_csv('random_forest_data.csv')

temperature = random_forest_dt['Temperature']
rainfall = random_forest_dt['Rainfall']
wind_speed = random_forest_dt['Wind Speed']
did_bike = random_forest_dt['Bike']

weather_conditions = []
for i in range(len(temperature)):
    weather_conditions.append([temperature[i], rainfall[i], wind_speed[i]])

x = weather_conditions
y = did_bike

clf = DecisionTreeClassifier()
clf.fit(x, y)

weather_conditions_test = [[22, 15, 10], [31, 8, 12], [6, 30, 5], [33, 40, 60]]

should_bike = clf.predict(weather_conditions_test)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

print('Should bike', should_bike)
Image(graph.create_png())
graph.write_jpg('C:\\Users\\filip\\ws\\Python\\Img\\img.jpg')
