import numpy as np
import matplotlib.pyplot as plt

mode = 0
train_coarse, train_fine, val_coarse, val_fine = [], [], [], []

file = open('losses.txt', 'r')
for line in file:
    line = line.split()
    if mode == 0:
        train_coarse.append(float(line[4]))
        train_fine.append(float(line[5]))
    else:
        val_coarse.append(float(line[4]))
        val_fine.append(float(line[5]))
    mode = (mode + 1) % 2
train_coarse, train_fine, val_coarse, val_fine = np.array(train_coarse), np.array(train_fine), np.array(val_coarse), np.array(val_fine)

plt.figure()
plt.plot(train_coarse, color = 'r')
plt.plot(train_fine, color = 'b')
plt.plot(val_coarse, color = 'g')
plt.plot(val_fine, color = 'k')
plt.legend(['Train Coarse', 'Train Fine', 'Validation Coarse', 'Validation Fine'])
plt.xlabel('Epochs (1000 iters)')
plt.ylabel('Loss')
plt.savefig('Loss.jpg')

plt.figure()
plt.plot(20 * np.log10(1 / np.sqrt(train_coarse)), color = 'r')
plt.plot(20 * np.log10(1 / np.sqrt(train_fine)), color = 'b')
plt.plot(20 * np.log10(1 / np.sqrt(val_coarse)), color = 'g')
plt.plot(20 * np.log10(1 / np.sqrt(val_fine)), color = 'k')
plt.legend(['Train Coarse', 'Train Fine', 'Validation Coarse', 'Validation Fine'])
plt.xlabel('Epochs (1000 iters)')
plt.ylabel('PSNR')
plt.savefig('PSNR.jpg')
