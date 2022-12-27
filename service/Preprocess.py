import pandas as pd


class Preprocess:

    def __int__(self, df):
        self.df = df

    def create_date_columns(self):
        print(self.df.head())
