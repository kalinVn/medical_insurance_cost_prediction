import pandas as pd
import numpy as np
import config

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from factory.Classifier import Classifier


class MedicalInsurancePrediction:

    def __init__(self):
        self.standard_data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        csv_path = config.CSV_PATH
        self.dataset = pd.read_csv(csv_path)

        self.scaler = StandardScaler()
        self.classifier_factory = Classifier()
        self.model = self.classifier_factory.get_model()

    def get_dataset(self):
        return self.dataset

    def preprocess(self):
        self.dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
        self.dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
        self.dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

        self.x = self.dataset.drop(columns='charges', axis=1)
        self.y = self.dataset['charges']

    def build(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=2)
        self.model.fit(self.x, self.y)

    def test_accuracy_score(self):
        x_train_prediction = self.model.predict(self.x_train)
        training_data_accuracy = metrics.r2_score(self.y_train, x_train_prediction)
        print("Accuracy on training data: ", training_data_accuracy)

        x_test_prediction = self.model.predict(self.x_test)
        test_data_accuracy = metrics.r2_score(self.y_test, x_test_prediction)
        print("Accuracy on test data: ", test_data_accuracy)

    def predict(self, data):
        inputs = np.array([data])
        columns = list(self.dataset.columns)[:-1]
        df = pd.DataFrame(inputs, columns=columns)

        prediction = self.model.predict(df)

        print('Insurance cost is USD: ', prediction[0])

