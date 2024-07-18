# import function : data from outside
from sklearn import datasets
# import function : using for splitting data
from sklearn.model_selection import train_test_split
# import function : using for data Standardization
from sklearn.preprocessing import StandardScaler
# import fucntion : using for Logistic Regression
from sklearn.linear_model import LogisticRegression

# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

# data standardization
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# model training
model = LogisticRegression()
model.fit(X_train_std, y_train)

# evaluate model
print('test accuracy: %.3f' % model.score(X_test_std, y_test))