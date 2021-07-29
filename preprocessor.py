from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self, df_features):
        self.y = None
        self.X = None
        self.df_features = df_features
        self.categorical = ['131', '133', '15', '155', '193', '196',
                            '216', '218', '220', '23', '24', '28', '29', '30', '95']

        self.feature_selection_list = [
            'vas_id', 'buy_time', '226', '1', '191', '207', '247', '164', '241', '193_mean',
            '144', '240', '3', '61', '145', '238', '114', '52', '128', '58', '53', '146',
            '229', '172', '115', '230', '64', '188', '243', '239', '222', '47', '127', '37',
            '39', '63', '5', '49', '210', '126', '43', '0', '224', '60', '248', '236', '187',
            '244', '211', '62', '130', '246', '234', '40', '143', '208', '111', '213', '20', '55 '
        ]

        self.cat_dict = {}

    def fit(self, X, y):
        self.X = X
        self.y = y

        return self

    def transform(self, X):

        self.df_features.drop_duplicates('id', keep='first', inplace=True)
        if 'Unnamed: 0' in self.df_features:
            self.df_features.drop(columns='Unnamed: 0', inplace=True)

        if 'Unnamed: 0' in X.columns:
            X.drop(columns='Unnamed: 0', inplace=True)
        X = pd.merge(X, self.df_features, how='left', on='id')

        X['merge_flag'] = 0
        X.loc[X['buy_time_x'] ==
              X['buy_time_y'], 'merge_flag'] = 1
        X.rename(columns={'buy_time_x': 'buy_time'}, inplace=True)
        X.drop(columns='buy_time_y', inplace=True)

        # drop const col =================================================

        X.drop(columns=['139', '203', '75', '81', '85'],
               inplace=True)

        # cat-> target_mean ==========================================

        if self.cat_dict == {}:
            X = pd.concat([X, self.y], axis=1)
            for col in self.categorical:
                temp = X.groupby(by=col)['target'].mean()
                temp.name = f'{col}_mean'
                self.cat_dict[col] = temp

        for dcol in self.cat_dict:
            X = pd.merge(X, self.cat_dict[dcol], how='left', on=dcol)
            X.drop(columns=dcol, inplace=True)

        if 'target' in X.columns:
            X.drop(columns='target', inplace=True)

        X['vas_id'] = X['vas_id'].astype('int8')
        X['buy_time'] = X['buy_time'].astype('int32')
        X.set_index('id', inplace=True)

        return X[self.feature_selection_list]
