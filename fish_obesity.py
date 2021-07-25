import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, confusion_matrix

fish_obesity_data_frame = pd.read_csv('fish obesity.csv')

# print(fish_obesity_data_frame.head())

labels = fish_obesity_data_frame['Obese']

features = fish_obesity_data_frame[['Height','Weight']]

# for i in range(len(fish_obesity_data_frame['Weight'])):
#     features.append([fish_obesity_data_frame['Weight'][i], fish_obesity_data_frame['Height'][i]])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# print(len(features))
# print(len(X_test))
# print(len(X_train))

clf = svm.SVC(C=1)
clf.fit(X_train, y_train)

# print(clf.predict([[96, 174]]))
# print(clf.score(X_test, y_test))

y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
# plt.show()
print(classification_report(y_test, y_pred))

# Acurácia, ou: em geral, qual o percentual de acerto?
accuracy = (tp + tn) / (tp + tn + fp + fn)
print ("accuracy", accuracy)

#Precision, ou: entre os que previ serem positivos, qual o percentual de acerto?
precision = tp / (tp + fp)
print ("precision", precision)

#Recall, ou: entre os que são positivos, qual o percentual de acerto?
recall = tp / (tp + fn)
print ("recall", recall)

#f1 score: media harmônica de Precision e Recall
f1_score = 2 * ((precision * recall) / (precision + recall))
print ("f1_score", f1_score)

# PLOT SVM
x = X_train['Weight']
y = X_train['Height']
label = y_train

plt.figure(figsize=(20, 10), dpi=120)

plt.scatter(x, y, c=label, cmap='viridis')

# The code below was copied from: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
x = np.linspace(xlim[0], xlim[1], 30)
y = np.linspace(ylim[0], ylim[1], 30)
Y, X = np.meshgrid(y, x)
xy = np.vstack([X.ravel(), Y.ravel()]).T
P = clf.decision_function(xy).reshape(X.shape)

# plot decision boundary and margins
ax.contour(X, Y, P, colors='k',
            levels=[-0.05, 0, 0.05], alpha=0.5,
            linestyles=['--', '-', '--'])
plt.show()