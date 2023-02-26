import pandas as pd
import numpy as np
import config
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics


class MedicalInsurancePredictionKeras:

    def __init__(self):
        self.standard_data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.history = None

        csv_path = config.CSV_PATH
        self.dataset = pd.read_csv(csv_path)

        self.scaler = StandardScaler()

        self.model = tf.keras.Sequential([
              tf.keras.layers.Dense(400, activation="relu"),
              tf.keras.layers.Dense(100, activation="relu"),
              tf.keras.layers.Dense(100, activation="relu"),
              tf.keras.layers.Dense(100, activation="relu"),
              tf.keras.layers.Dense(1)
        ])

    def get_dataset(self):
        return self.dataset

    def get_history(self):
        return self.history

    def preprocess(self):
        self.dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
        self.dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
        self.dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

        self.x = self.dataset.drop(columns='charges', axis=1)
        self.y = self.dataset['charges']

    def build(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=2)
        self.model.compile(loss=tf.keras.losses.mae,
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics=['mae'])
        self.history = self.model.fit(self.x, self.y, epochs=200, verbose=0)
        # self.model.evaluate(self.x_test, self.y_test)

    def test_accuracy_score(self):
        x_train_prediction = self.model.predict(self.x_train)
        training_data_accuracy = metrics.r2_score(self.y_train, x_train_prediction)
        print("Accuracy on training data: ", training_data_accuracy)

        x_test_prediction = self.model.predict(self.x_test)
        test_data_accuracy = metrics.r2_score(self.y_test, x_test_prediction)
        print("Accuracy on test data: ", test_data_accuracy)


    def predict(self, data):
        prediction = self.model.predict([data])

        print('Insurance cost is USD: ', prediction[0][0])


