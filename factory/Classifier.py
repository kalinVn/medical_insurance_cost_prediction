from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import config




class Classifier:

    def __init__(self):
        self.name = config.MODEL

    def get_model(self):
        if self.name == "svm":
            return svm.SVC(kernel='linear')
        elif self.name == "logistic_regression":
            return LinearRegression(solver='lbfgs', max_iter=100000)
        elif self.name == "linear_regression":
            return LinearRegression()
        else:
            return Lasso()

