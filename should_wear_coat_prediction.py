from sklearn.linear_model import LogisticRegression

temperature = [[30], [12], [14], [18], [25], [5], [15], [27], [1]]
coat = [False, True, True, True, False, True, True, False, False]
should_wear_coat = [[1], [12], [18], [120], [25]]
clf = LogisticRegression(random_state=0).fit(temperature, coat)

print(clf.predict(should_wear_coat))
print(clf.score(temperature, coat))