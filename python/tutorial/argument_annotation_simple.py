'''
Created on 5 May 2015

@author: edwin
'''
import logging
logging.basicConfig(level=logging.DEBUG)
import ibcc
import numpy as np
import matplotlib.pyplot as plt

# LOAD CROWDSOURCED DATA ----------------------------------------------

datafile = "./data/crowd_matrix_arguments_nonans.csv"

Ctable_str = np.genfromtxt(datafile, dtype=str, delimiter=',')

# convert strings to floats
ulabels = np.unique(Ctable_str)
Ctable = np.zeros(Ctable_str.shape, dtype=float)
counter = 0
for l in ulabels:
    if not len(l) or l=='nan' or l=='NaN' or l=='NAN':
        counter += 1
        Ctable[Ctable_str==l] = np.NaN
    else:
        Ctable[Ctable_str==l] = counter
    counter += 1  
    
# RUN IBCC --------------------------------------------------------------
    
nclasses = 6
nworkers = Ctable.shape[1]

alpha0 = np.ones((nclasses, nclasses, nworkers))
alpha0[np.arange(nclasses),np.arange(nclasses),:] += 1.0
nu0 = np.ones(nclasses, dtype=float) + 100
combiner = ibcc.IBCC(nclasses=nclasses, nscores=nclasses, alpha0=alpha0, nu0=nu0)
probabilities = combiner.combine_classifications(Ctable, table_format=True) # returns the probabilities for each class
predicted_labels = np.argmax(probabilities, axis=1) # chooses the class with highest probability as best guess annotation

# SAVE RESULTS TO CSV FILE --------------------------------------------------
np.savetxt('./output/arguments_pyibcc_probabilities.csv', probabilities, delimiter=',', fmt='%.2f')
np.savetxt('./output/arguments_pyibcc_predicted_labels.csv', predicted_labels, delimiter=',', fmt='%.f')

# PLOT CONFUSION MATRIX ----------------------------------------------------
from scipy.stats import beta
plt.figure()
# for k in range(combiner.alpha.shape[2]):
k = 0 # worker ID to plot
alpha_k  = combiner.alpha[:, :, k]
pi_k = alpha_k / np.sum(alpha_k, axis=1)[:, np.newaxis]
print "Expected confusion matrix for worker %i" % k
print pi_k
    
x = np.arange(20) / 20.0
for j in range(alpha_k.shape[0]):
    pdfj = beta.pdf(x, alpha_k[j, j], np.sum(alpha_k[j, :]) - alpha_k[j,j] )
    plt.plot(x, pdfj, label='True class %i' % j)
plt.legend(loc='best')
plt.ylabel('density')
plt.xlabel('p(correct annotation)')