import matplotlib.pyplot as plt
from data_processing import DataPreparation
from Tuning import HyperparameterTuning
from train_selected_model import TrainSelectedModel

myclass = DataPreparation("C:\\Users\\steph\\Desktop\\final_code_thesis\\Data")
myclass.create_dataframe_for_every_file()
combined_X_train, combined_X_test, combined_y_train, combined_y_test = myclass.split_train_test_for_every_df("EoS_construction_reverse")
print(combined_X_test,combined_y_test)
print(combined_X_train,combined_y_train)
my_class_2 = HyperparameterTuning(combined_X_train,combined_y_train,combined_X_test,combined_y_test)
best_params = my_class_2.run_tuning()
print(best_params)
my_class_3 = TrainSelectedModel(best_params,combined_X_train,combined_y_train,combined_X_test,combined_y_test,"ANN_EoS_construction_reverse_output")
history,predictions = my_class_3.train_model()
print(history)

fig_1 = plt.figure()
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

fig_2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')


plt.tight_layout()

plt.show()