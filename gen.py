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

Precision_eca = []
Recall_eca = []
mAP_eca = []
F1_eca = []

Precision_GhostM = []
Recall_GhostM = []
mAP_GhostM = []
F1_GhostM = []

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

def readTxtGhostM(fileDir):
    with open(fileDir, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()
            Precision_GhostM.append(float(line[8]))
            Recall_GhostM.append(float(line[9]))
            mAP_GhostM.append(float(line[10]))
            F1_GhostM.append(float(line[11]))

def readTxtECA(fileDir):
    with open(fileDir, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()
            Precision_eca.append(float(line[8]))
            Recall_eca.append(float(line[9]))
            mAP_eca.append(float(line[10]))
            F1_eca.append(float(line[11]))

readTxtGhost('F:/DeepLearning/yolov3/resultsghost.txt')
readTxt('F:/DeepLearning/yolov3/results_raw.txt')
readTxtGhostM('F:/DeepLearning/yolov3/results.txt')
readTxtECA('F:/DeepLearning/yolov3/resultsse.txt')

plt.figure(1)
plt.plot(epoch, Precision, linewidth=0.5, color='r', label='Raw')
plt.plot(epoch, Precision_Ghost, linewidth=0.5, color='b', label='GM')
plt.plot(epoch, Precision_GhostM, linewidth=0.5, color='y', label='GB')
plt.plot(epoch, Precision_eca, linewidth=0.5, color='g', label='GB_ECA')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('')
plt.legend()

plt.savefig('pre.png')

plt.figure(2)
plt.plot(epoch, Recall, linewidth=0.5, color='r', label='Raw')
plt.plot(epoch, Recall_Ghost, linewidth=0.5, color='b', label='GM')
plt.plot(epoch, Recall_GhostM, linewidth=0.5, color='y', label='GB')
plt.plot(epoch, Recall_eca, linewidth=0.5, color='g', label='GB_ECA')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.savefig('recall.png')

plt.figure(3)
plt.plot(epoch, mAP, linewidth=0.5, color='r', label='Raw')
plt.plot(epoch, mAP_Ghost, linewidth=0.5, color='b', label='GM')
plt.plot(epoch, mAP_GhostM, linewidth=0.5, color='y', label='GB')
plt.plot(epoch, mAP_eca, linewidth=0.5, color='g', label='GB_ECA')
plt.xlabel('Epochs')
plt.ylabel('mAP')
plt.legend()

plt.savefig('mAP.png')


plt.figure(4)
plt.plot(epoch, F1, linewidth=0.5, color='r', label='Raw')
plt.plot(epoch, F1_Ghost, linewidth=0.5, color='b', label='GM')
plt.plot(epoch, F1_GhostM, linewidth=0.5, color='y', label='GB')
plt.plot(epoch, F1_eca, linewidth=0.5, color='g', label='GB_ECA')
plt.xlabel('Epochs')
plt.ylabel('F1')
plt.legend()

plt.savefig('f1.png')
