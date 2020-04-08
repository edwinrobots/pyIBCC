import numpy as np

print('Configuring IBCC')

scores = np.array([0, 1, 2, 3])
nScores = len(scores)
nClasses = 4
nu0 = np.array([30, 30, 30, 10])
alpha0 = np.array([[2,1,1,1], [1,2,1,1], [1,1,2,1], [1,1,1,2]])
# has also to be specified in ibccdata.py
inputFile =   '../data/metaphor_annotation/scale_hits_IBCC_input.csv'
#goldFile =    './data/test1/gold.csv'
# has also to be specified in ibccdata.py
outputFile =  '../output/output_for_scale_hit.csv'
confMatFile = '../output/confMat_for_scale_hit.csv'
