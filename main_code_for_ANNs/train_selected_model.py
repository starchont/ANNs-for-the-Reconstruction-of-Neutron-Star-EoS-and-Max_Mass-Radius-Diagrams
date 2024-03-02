import pickle
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.callbacks import EarlyStopping


class TrainSelectedModel:

    def __init__(self, best_params, X_train, y_train, X_test, y_test, name_for_model):
        self.best_params = best_params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.name_for_model = name_for_model

    def train_model(self):
        selected_model = Sequential()
        selected_model.add(Input(shape=(self.X_train.shape[1],)))
        for layer_i in range(self.best_params['num_layers']):
            selected_model.add(Dense(self.best_params["n_units_layer" + str(layer_i)],
                                     activation=self.best_params['activ_func_layer' + str(layer_i)]))
        selected_model.add(Dense(self.y_train.shape[1], ))
        print(selected_model.summary())
        selected_model.compile(
            loss="mae",
            optimizer='adam',
            metrics=["mse"]
        )
        history = selected_model.fit(self.X_train, self.y_train, epochs=25000, batch_size=self.best_params["batch_size"],validation_data=(self.X_test, self.y_test),callbacks=EarlyStopping(monitor=('loss'), patience=300))

        selected_model.save(self.name_for_model+'.h5')

        with open(self.name_for_model + "_datasetSplit.pkl", 'wb') as file:
            pickle.dump((self.X_train, self.X_test, self.y_train, self.y_test), file)

        predictions = selected_model.predict(self.X_test)
        return [history, predictions]
