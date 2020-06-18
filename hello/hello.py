from sklearn import tree


my_tree = tree.DecisionTreeClassifier()
dectrung = [[1, 3, 3, 7],
            [5, 2, 4, 6],
            [1, 2, 4, 6],
            [5, 4, 4, 3],
            [3, 2, 3, 7],
            [5, 2, 7, 7],
            [3, 3, 3, 6],
            [1, 4, 4, 7],
            ]

nhan = [0, 1, 1, 0, 0, 0, 0, 1]

result = my_tree.fit(dectrung, nhan)

kq = result.predict([[1, 3, 4,6]])
print(kq)