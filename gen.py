import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.size": 13,
}

rcParams.update(config)

epoch = []

for i in range(100):
    epoch.append(i + 1)

Precision = []
Recall = []
mAP = []
F1 = []

Precision_Ghost = []
Recall_Ghost = []
mAP_Ghost = []
F1_Ghost = []

def readTxt(fileDir):
    with open(fileDir, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()
            Precision.append(float(line[8]))
            Recall.append(float(line[9]))
            mAP.append(float(line[10]))
            F1.append(float(line[11]))

def readTxtGhost(fileDir):
    with open(fileDir, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()
            Precision_Ghost.append(float(line[8]))
            Recall_Ghost.append(float(line[9]))
            mAP_Ghost.append(float(line[10]))
            F1_Ghost.append(float(line[11]))

readTxtGhost('F:/DeepLearning/yolov3/resultsghost.txt')
readTxt('F:/DeepLearning/yolov3/results.txt')

plt.figure(1)
plt.plot(epoch, Precision, linewidth=0.5, color='r', label='raw')
plt.plot(epoch, Precision_Ghost, linewidth=0.5, color='b', label='ghost')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('')
plt.legend()

plt.savefig('raw_ghost_pre.png')

plt.figure(2)
plt.plot(epoch, Recall, linewidth=0.5, color='r', label='raw')
plt.plot(epoch, Recall_Ghost, linewidth=0.5, color='b', label='ghost')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.savefig('raw_ghost_recall.png')

plt.figure(3)
plt.plot(epoch, mAP, linewidth=0.5, color='r', label='raw')
plt.plot(epoch, mAP_Ghost, linewidth=0.5, color='b', label='ghost')
plt.xlabel('Epochs')
plt.ylabel('mAP')
plt.legend()

plt.savefig('raw_ghost_mAP.png')


plt.figure(4)
plt.plot(epoch, F1, linewidth=0.5, color='r', label='raw')
plt.plot(epoch, F1_Ghost, linewidth=0.5, color='b', label='ghost')
plt.xlabel('Epochs')
plt.ylabel('F1')
plt.legend()

plt.savefig('raw_ghost_f1.png')
