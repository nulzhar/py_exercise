import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix

random_forest_dt = pd.read_csv('banknote_data.csv')

# Método 1 para gerar o features
#variance = random_forest_dt['Variance']
#skewness = random_forest_dt['Skewness']
#curtosis = random_forest_dt['Curtosis']
#entropy = random_forest_dt['Entropy']

#features = []
#for i in range(len(variance)):
#    features.append([variance[i], skewness[i], curtosis[i], entropy[i]])

# Método 2 para gerar o features
features = random_forest_dt[['Variance', 'Skewness', 'Curtosis', 'Entropy']].values

labels = random_forest_dt['Class']

# Split the data in test and train
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#print(features)
#print(X_train)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, values_format='.0f')
plt.show()
