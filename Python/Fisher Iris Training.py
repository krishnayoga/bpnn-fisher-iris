import pandas as pd
import numpy as np
from numpy import *
import torch
import matplotlib.pyplot as plt
import time

data_length = 150

#pakai pandas
data_input = pd.read_csv('iris_input.csv', header = None)
data_target = pd.read_csv('iris_target.csv', header = None)
print ('Data input\n',data_input,'\nData target\n',data_target)

#ubah ke numpy (dalam matriks)
data_input_np = np.c_[data_input]
data_target_np = np.c_[data_target]
print('Data input dalam matriks\n',data_input_np,'\nData target dalam matriks\n',data_target_np)

#siap-siap normalisasi data
#masukkan tiap data ke masing-masing variabel
data_sl = data_input_np[:data_length,[0]]
data_sw = data_input_np[:data_length,[1]]
data_pl = data_input_np[:data_length,[2]]
data_pw = data_input_np[:data_length,[3]]

#cek hasil pemindahan
print('Data sepal length\n', data_sl)
print('\nData sepal width\n', data_sw)
print('\nData petal length\n', data_pl)
print('\nData petal width\n', data_pw)

#cari maksimum
max_sl = max(data_sl)
max_sw = max(data_sw)
max_pl = max(data_pl)
max_pw = max(data_pw)

#cari minimum
min_sl = min(data_sl)
min_sw = min(data_sw)
min_pl = min(data_pl)
min_pw = min(data_pw)

#normalisasi sepal length
sl_norm = np.zeros(shape=(data_length,1))

for k in range(data_length):
    sl_norm[k,:] = (data_sl[k] - min_sl) / (max_sl - min_sl)

print('sl_norm\n',sl_norm)

#normalisasi sepal width
sw_norm = np.zeros(shape=(data_length,1))

for k in range(data_length):
    sw_norm[k,:] = (data_sw[k] - min_sw) / (max_sw - min_sw)

print('sw_norm\n',sw_norm)

#normalisasi petal length
pl_norm = np.zeros(shape=(data_length,1))

for k in range(data_length):
    pl_norm[k,:] = (data_pl[k] - min_pl) / (max_pl - min_pl)

print('pl_norm\n',pl_norm)

#normalisasi petal width
pw_norm = np.zeros(shape=(data_length,1))

for k in range(data_length):
    pw_norm[k,:] = (data_pw[k] - min_pw) / (max_pw - min_pw)

print('pw_norm\n',pw_norm)

#gabungin data input
data_input_ok = np.hstack((sl_norm,sw_norm,pl_norm,pw_norm))
data_target_ok = data_target_np

print('data input\n',data_input_ok)
print('data target\n',data_target_ok)

#save ke csv
np.savetxt("matriks_input.csv",data_input_ok,delimiter=",")
np.savetxt("matriks_target.csv",data_target_ok,delimiter=",")

#inisialisasi parameter
batch_size = 150
input_dimension = 4
hidden_unit = 24
output_dimension = 3
epoch_max = 1000
loss_plot = np.zeros(shape=(epoch_max,1))
t_plot = np.zeros(shape=(epoch_max,1))
error_now = 1000
target_error = 0.0001
i = 0

#ubah ke tensor
data_input_tensor = torch.from_numpy(data_input_ok).float()
data_target_tensor = torch.from_numpy(data_target_ok).float()

#inisialisasi model
#model = TwoLayerNet(input_dimension, hidden_unit, output_dimension)
model = torch.nn.Sequential(
    torch.nn.Linear(input_dimension, hidden_unit),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_unit, output_dimension)
)

loss_fcn = torch.nn.MSELoss()

learning_rate = 0.1
#for t in range(epoch_max):
while error_now > target_error and i < epoch_max:
#while i < epoch_max:
#while error_now > target_error:
    y_pred = model(data_input_tensor)
    
    loss = loss_fcn(y_pred,data_target_tensor)
    if i % 10 == 0:
        print(i, loss.item())
        
        
    loss_plot[i] = loss.item()
    t_plot[i] = i
    print(i,loss_plot[i],t_plot[i])
    
    log_weight = list(model.parameters())
    model.zero_grad()
    
    error_now = loss.item()
    
    if i % 50 == 0:
        torch.save(model,'training_f_{}.pt'.format(int(time.time())))

    i += 1
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

#loss_plot_numpy = loss_plot.numpy()
plt.plot(t_plot,loss_plot)
plt.show()

#print(log_weight)
