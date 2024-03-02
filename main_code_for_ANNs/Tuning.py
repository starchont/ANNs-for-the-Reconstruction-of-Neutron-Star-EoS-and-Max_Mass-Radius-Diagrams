import optuna
from keras.models import Sequential
from keras.layers import Dense, Input

class HyperparameterTuning:
    def __init__(self,X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run_tuning(self):
        def objective_for_ANN(trial):
            model = Sequential()
            model.add(Input(shape=(self.X_train.shape[1],)))
            num_layers = trial.suggest_int("num_layers", low=0, high=5)
            for layer_i in range(num_layers):
                n_units = trial.suggest_categorical(f"n_units_layer{layer_i}",[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
                actv_function = trial.suggest_categorical(f'activ_func_layer{layer_i}',['relu', 'tanh', 'elu', 'sigmoid', 'selu'])
                model.add(Dense(n_units, activation=actv_function))
            model.add(Dense(2, ))
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            model.compile(loss="mean_squared_error",optimizer='adam',metrics=["mean_absolute_error"])
            model.fit(self.X_train,self.y_train,shuffle=True,batch_size=batch_size,epochs=20,verbose=False,)
            score = model.evaluate(self.X_test, self.y_test, verbose=0)
            return score[0]

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective_for_ANN, n_trials=10)
        trial = study.best_trial
        best_params = study.best_params
        return best_params
