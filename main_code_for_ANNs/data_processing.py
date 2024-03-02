import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreparation:
    def __init__(self, path):
        self.path = path
        self.dataframe_list = []
        self.create_dataframe_for_every_file()

    def add_equation(self):
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            if not os.path.isfile(file_path) or not filename.lower().endswith('.txt'):
                continue
            try:
                equation = os.path.splitext(filename)[0]
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    lines_to_add = [lines[0].strip()+', '+'EoS'+'\n']
                    for line in lines[1:]:
                        lines_to_add.append(line.strip()+' '+equation+'\n')
                with open(file_path, 'w') as file:
                    file.writelines(lines_to_add)
            except Exception as e:
                print(f'An Error occured while reading "{file_path}" : {e}')

    def create_dataframe_for_every_file(self):
        for filename in os.listdir(self.path):
            if filename.endswith(".txt"):
                with open(filename,'r') as file:
                    headers = file.readline().strip().split()
                    headers = [header.replace(',', ' ') for header in headers]
                    headers.pop(0)
                df = pd.read_csv(filename, delimiter=' ', names=headers, skiprows=1)
                df = df[(df.iloc[:, 4].astype(float) >= float(8.0)) & (df.iloc[:, 4].astype(float) <= float(20.0))]
                df = df.dropna()
                #print(df)
                self.dataframe_list.append(df)

    @staticmethod
    def define_X_and_Y(df, type_of_ANN):
        df['E '] = df['E '].astype(float)
        df['Pc '] = df['Pc '].astype(float)
        df['M '] = df['M '].astype(float)
        df['R '] = df['R '].astype(float)
        if type_of_ANN == 'EP':
            X = df.loc[:, 'M ':'R '].values.astype(float).reshape(-1, 2)
            X = StandardScaler().fit_transform(X)
            y = df.loc[:, 'Pc ':'E '].values.astype(float).reshape(-1, 2)
        elif type_of_ANN == 'MR':
            X = df.loc[:, 'Pc ':'E '].values.astype(float).reshape(-1, 2)
            X = StandardScaler().fit_transform(X)
            y = df.loc[:, 'M ':'R '].values.astype(float).reshape(-1, 2)

        return X, y

    @staticmethod
    def split_train_test(X, y):
        return train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    @staticmethod
    def concatenate_dataframes(combined_X_train, combined_y_train, X_train_df, y_train_df):
        combined_X_train = pd.concat([combined_X_train, X_train_df], ignore_index=True)
        combined_y_train = pd.concat([combined_y_train, y_train_df], ignore_index=True)
        return combined_X_train, combined_y_train

    @staticmethod
    def shuffle_dataframes(combined_X, combined_y):
        num_rows = combined_X.shape[0]
        random_indices = np.random.default_rng(seed=42).permutation(num_rows)
        combined_X_shuffle = combined_X.iloc[random_indices].reset_index(drop=True)
        combined_y_shuffle = combined_y.iloc[random_indices].reset_index(drop=True)
        return combined_X_shuffle, combined_y_shuffle

    def split_train_test_for_every_df(self, type_of_ANN):
        combined_X_train = pd.DataFrame()
        combined_y_train = pd.DataFrame()
        combined_X_test = pd.DataFrame()
        combined_y_test = pd.DataFrame()

        for df in self.dataframe_list:
            X, y = self.define_X_and_Y(df, type_of_ANN)
            X_train, X_test, y_train, y_test = self.split_train_test(X, y)

            if type_of_ANN == "MR":
                X_train_df = pd.DataFrame(X_train, columns=['Pc',"E"])
                y_train_df = pd.DataFrame(y_train, columns=['M', 'R'])
                X_test_df = pd.DataFrame(X_test, columns=['Pc',"E"])
                y_test_df = pd.DataFrame(y_test, columns=['M', 'R'])
            elif type_of_ANN == "EP":
                X_train_df = pd.DataFrame(X_train, columns=['M', 'R'])
                y_train_df = pd.DataFrame(y_train, columns=['Pc',"E"])
                X_test_df = pd.DataFrame(X_test, columns=['M', 'R'])
                y_test_df = pd.DataFrame(y_test, columns=['Pc',"E"])

            combined_X_train, combined_y_train = self.concatenate_dataframes(combined_X_train, combined_y_train,
                                                                             X_train_df, y_train_df)
            combined_X_test, combined_y_test = self.concatenate_dataframes(combined_X_test, combined_y_test, X_test_df,
                                                                           y_test_df)

        combined_X_train_shuffle, combined_y_train_shuffle = self.shuffle_dataframes(combined_X_train, combined_y_train)
        combined_X_test_shuffle, combined_y_test_shuffle = self.shuffle_dataframes(combined_X_test, combined_y_test)

        return combined_X_train_shuffle, combined_X_test_shuffle, combined_y_train_shuffle, combined_y_test_shuffle
