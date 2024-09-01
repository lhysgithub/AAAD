import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd

stream1 = pd.read_csv("stream1.csv",header=None).iloc[:,1].to_numpy()
stream2 = pd.read_csv("stream2.csv",header=None).iloc[:,1].to_numpy()
stream3 = pd.read_csv("stream3.csv",header=None).iloc[:,1].to_numpy()
x = np.arange(len(stream1))

plt.subplot(3,1,1)
plt.subplots_adjust(left=0.04,right=0.98,top=0.98,bottom=0.1) # ,wspace=0.1,hspace=0.1
plt.plot(x, stream1, "p-c", label = "stream1", alpha=0.7)
plt.legend()
plt.subplot(3,1,2)
plt.plot(x, stream3, "p-c", label = "stream2", alpha=0.7)
plt.legend()
plt.subplot(3,1,3)
plt.plot(x, stream2, "p-c", label = "stream3", alpha=0.7)
plt.legend()
plt.xlabel("Timestamp")
plt.savefig("../analysis2/aaad_mo.pdf")

