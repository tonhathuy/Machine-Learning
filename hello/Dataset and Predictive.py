from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
iris_dataset = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)
# print(Y_test)

model = DecisionTreeClassifier() 

myModel = model.fit(X_train, Y_train)

X_new = np.array([[6.0, 3.23, 4.5, 2.0]])
# print(myModel.predict(X_test)) 
# print(myModel.predict(X_new))
print(myModel.score(X_test,Y_test))