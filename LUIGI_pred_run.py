import pandas as pd
import os
from pathlib import Path
from preprocessor import Preprocessing
import luigi
import pickle


class PredictionPipeline(luigi.Task):
    TRAIN_FILENAME = luigi.Parameter()
    TEST_FILENAME = luigi.Parameter()
    FEATURES_FILENAME = luigi.Parameter()
    MODEL_FILENAME = luigi.Parameter()

    def run(self):
        df_train = pd.read_csv(self.TRAIN_FILENAME)
        df_test = pd.read_csv(self.TEST_FILENAME)
        df_features = pd.read_csv(self.FEATURES_FILENAME)
        X = df_train.drop(columns='target')
        y = df_train['target']
        prep = Preprocessing(df_features)
        prep.fit(X, y)
        X_test = prep.transform(df_test)

        with open(self.MODEL_FILENAME, 'rb') as input_file:
            model = pickle.load(input_file)
        predictions = model.predict(X_test)
        X_test['target'] = predictions[:, 1]
        X_test.set_index('id', inplace=True)
        X_test.to_csv('answers_test.csv')

    def output(self):
        return luigi.LocalTarget('answers_test.csv')


if __name__ == '__main__':  # из условия что test в корне(так в задании) , а train и features в папке data
    TRAIN_FILENAME = os.path.join(Path(os.getcwd()), "data", "data_train.csv")
    TEST_FILENAME = 'data_test.csv'
    FEATURES_FILENAME = os.path.join(Path(os.getcwd()), "data", "features.csv")
    MODEL_FILENAME = os.path.join(Path(os.getcwd()), "model", "xgb_model.pickle")

    luigi.build([PredictionPipeline(TRAIN_FILENAME, TEST_FILENAME, FEATURES_FILENAME, MODEL_FILENAME)])
