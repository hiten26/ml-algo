import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNN():
    def __init__(self, k) -> None:
        self.k = k
        self.x_train = None
        self.y_train = None
    
    def eucleadian(self, x, y):
        return np.sqrt(np.sum((x-y)**2))
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def predict(self, x):
        preds = [self._predict(y) for y in x]
        return preds
    
    def _predict(self, x):
        #copmute the distance
        distances = [self.eucleadian(x, x_train) for x_train in self.x_train]

        #get the nearest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[ind] for ind in k_indices]

        #majority vote
        most_common = Counter(k_nearest_labels).most_common()

        return most_common[0][0]

    

if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = KNN(5)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(preds)

    acc = np.sum(preds == y_test) / len(y_test)

    print("Accuracy: ", acc)