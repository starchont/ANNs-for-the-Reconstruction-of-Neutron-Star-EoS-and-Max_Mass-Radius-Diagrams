import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = os.path.join(os.getcwd())

file_list = []
for file_name in os.listdir(data_dir):
    if file_name.endswith(".txt"):
        file_list.append(file_name)

file_list_sorted = sorted(file_list)
half_len = len(file_list_sorted) // 2
first_half = file_list_sorted[:half_len]
second_half = file_list_sorted[half_len:]
print(first_half)
print(second_half)

for file_name in first_half:
    df = pd.read_csv(file_name, delimiter=' ')
    df = df.drop(0)
    print(df)
    plt.figure(0)
    plt.scatter(df.iloc[:,4], df.iloc[:,3], label=file_name.split('.txt')[0], s=1)
    plt.figure(2)
    plt.scatter(df.iloc[:,1], df.iloc[:,2], label=file_name.split('.txt')[0], s=1)

for file_name in second_half:
    df = pd.read_csv(file_name, delimiter=' ')
    df = df.drop(0)
    print(df)
    plt.figure(1)
    plt.scatter(df.iloc[:, 4], df.iloc[:, 3], label=file_name.split('.txt')[0], s=1)
    plt.figure(3)
    plt.scatter(df.iloc[:,1], df.iloc[:,2], label=file_name.split('.txt')[0], s=1)

plt.figure(0)
plt.legend()
plt.xlim(8, 20)
plt.xlabel("R(km)")
plt.ylabel("M(M_sun)")


plt.figure(1)
plt.legend()
plt.xlim(8, 20)
plt.xlabel("R(km)")
plt.ylabel("M(M_sun)")

plt.figure(2)
plt.legend()
#plt.xlim(8, 20)
plt.xlabel("Pc")
plt.ylabel("E")

plt.figure(3)
plt.legend()
#plt.xlim(8, 20)
plt.xlabel("Pc")
plt.ylabel("E")

plt.show()

