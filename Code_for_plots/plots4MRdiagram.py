import pickle
import matplotlib.pyplot as plt
from keras.models import load_model


model = load_model("ANN_MR_2nd_output.h5")
for layer in model.layers:
    print(f"Layer Name: {layer.name}, Activation Function: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else layer.activation}")
print(model.summary())

with open("ANN_MR_2nd_output_datasetSplit.pkl","rb") as file:
    X_train_EP, X_test_EP, y_train_EP, y_test_EP = pickle.load(file)

with open("ANN_MR_2nd_output_model_results.pkl","rb") as file:
    results_dict = pickle.load(file)

with open("your_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Inverse transform the scaled data
X_train_EP_original = scaler.inverse_transform(X_train_EP)
X_test_EP_original = scaler.inverse_transform(X_test_EP)
print(X_test_EP_original)
print(X_train_EP_original)

print(y_test_EP)
plt.figure(0)
plt.scatter(results_dict['predictions'][:,1],results_dict['predictions'][:,0], s=1)
plt.scatter(y_test_EP.iloc[:,1], y_test_EP.iloc[:,0], s=1)
plt.show()