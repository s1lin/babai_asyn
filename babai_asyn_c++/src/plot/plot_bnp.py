import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from matplotlib import rcParams

rc = {"font.family": "serif", "mathtext.fontset": "stix"}
legend_properties = {'weight': 'bold'}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 20})


def save_data(n, i, max_proc, min_proc, qrT, asplT, totalT, bnp, ber, itr):
    np.savez(f'./test_result/{n}_report_plot_{i}_BNP.npz', n=n, i=i, max_proc=max_proc, min_proc=min_proc,
             qrT=qrT, asplT=asplT, totalT=totalT, bnp=bnp, ber=ber, itr=itr)
