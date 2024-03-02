import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

model_EoS = load_model("ANN_EoS_construction_output.h5")
model_MR = load_model("ANN_MR_2nd_output.h5")
model_EoS_reverse = load_model("ANN_EoS_construction_reverse_output.h5")
model_MR_reverse = load_model("ANN_MR_reverse_output.h5")


data_Gw = np.loadtxt('data_GW.txt', delimiter=' ')
data_1 = data_Gw[:, [0, 4]]
data_2 = data_Gw[:, [1, 5]]
print(data_1.shape)
scaler = StandardScaler()

# Give MR get E
output_1 = model_MR_reverse.predict(data_1)
output_1_scaled = scaler.fit_transform(output_1[:,0].reshape(-1, 1)).flatten()

# Give E get Pc
output_2 = model_EoS_reverse.predict(output_1_scaled)
output_2_scaled = scaler.fit_transform(output_2[:,0].reshape(-1, 1)).flatten()

# Give pc get E
output_3 = model_EoS.predict(output_2_scaled)
output_3_scaled = scaler.fit_transform(output_3[:,0].reshape(-1, 1)).flatten()

# Give E get MR
output_4 = model_MR.predict(output_3_scaled)

output_1_2 = model_MR_reverse.predict(data_2)
output_1_scaled_2 = scaler.fit_transform(output_1_2[:,0].reshape(-1, 1)).flatten()

# Give E get Pc
output_2_2 = model_EoS_reverse.predict(output_1_scaled_2)
output_2_scaled_2 = scaler.fit_transform(output_2_2[:,0].reshape(-1, 1)).flatten()

# Give pc get E
output_3_2 = model_EoS.predict(output_2_scaled_2)
output_3_scaled_2 = scaler.fit_transform(output_3_2[:,0].reshape(-1, 1)).flatten()

# Give E get MR
output_4_2 = model_MR.predict(output_3_scaled_2)
print(output_4_2.shape)
print(output_1)
print(len(output_1), len(output_2), len(output_3))

plt.figure(0)
plt.scatter(data_Gw[:, -1], data_Gw[:, 1], s=0.5)
plt.scatter(data_Gw[:, 4], data_Gw[:, 0], s=0.5)
plt.scatter(output_4[:, 1], output_4[:, 0], s=2.5)
plt.scatter(output_4_2[:, 1], output_4_2[:, 0], s=2.5)
plt.xlabel('R(km)')
plt.ylabel('M[M_sun]')
plt.legend()

plt.figure(1)
plt.scatter(output_2[:, 0], output_3[:, 0], s=2.5)
plt.xlabel('Pc')
plt.ylabel('E')
plt.xlim(0,900)
plt.show()
