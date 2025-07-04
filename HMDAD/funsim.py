
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import obonet
import networkx as nx
import math


def PBPA(RNA_i, RNA_j, di_sim, rna_di):
    diseaseSet_i = rna_di[RNA_i] > 0
    diseaseSet_j = rna_di[RNA_j] > 0
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]
    ijshape = diseaseSim_ij.shape
    if ijshape[0] == 0 or ijshape[1] == 0:
        return 0
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])
def getRNASiNet(RNAlen, diSiNet, rna_di):
    RNASiNet = np.zeros((RNAlen, RNAlen))
    for i in range(RNAlen):
        for j in range(i+1, RNAlen):
            RNASiNet[i, j] = RNASiNet[j, i] = PBPA(i, j, diSiNet, rna_di)
    return RNASiNet


RNAlen=39
diSiNet=np.loadtxt('D:/Desktop文件/TASEMDA/dataset/HMDAD/GSM.csv', delimiter=',')

rna_di=np.loadtxt('D:/Desktop文件/TASEMDA/dataset/HMDAD/A.csv', delimiter=',')
rna_di=rna_di.T
RNASiNet=getRNASiNet(RNAlen, diSiNet, rna_di)
np.savetxt('./disfunsim.csv', RNASiNet, delimiter=',')