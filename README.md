# Using IBCC to Reduce Classifications and Produce User Weightings

IBCC is a probabilistic method for reducing a set of classifications from crowd workers, citizen scientists, automated classifiers, or other 
annotators to a combined classification. 
As part of the process of reduction, IBCC weights each worker with a confusion matrix. 
This page will describe how to run the IBCC code and what data format it requires, and how to interpret its results so you can compare IBCC with other reduction/combination/user weighting methods.

For more information about the method, please see the following:
1. "Bayesian combination of multiple, imperfect classifiers", Simpson, Roberts, Smith, Lintott (2011): www.orchid.ac.uk/eprints/7/4/vbibcc_workshop.pdf
1. The method is a variant of "Bayesian Classifier Combination" described by Kim and Ghahramani (2012): http://proceedings.mlr.press/v22/kim12/kim12.pdf

For anything that's unclear here, please ask edwin.simpson@gmail.com.

### Warning

While ibcc.py is fairly well tested, the repository contains other variants of the method that may be buggy or slow to run.

## Getting the software

All the python code is at: https://github.com/edwinrobots/pyIBCC/python. This contains fairly trivial examples of the input and output files, and an example configuration file.

## Input Data

IBCC requires as input the following CSV files.
 
### Input.csv

This is the input classifications table containing the responses from workers. Can be in one of two formats. 

a) Sparse list format (suitable when many classifiers provide only a few responses), which is a 2-D array of size Nx3 where N is the number of classifications. The sparse list has the following columns: userID, subjectID, score. All three columns should contain **integer** values. 

1. The userID should be a unique identifier for the worker. Users who were not logged in could all be assigned the same ID if you want to merge them, or assigned some other arbitrary ID value if you wish to distinguish them by session. 

1. The subjectID is an identifier for the target subjects we wish to classify (sometimes called asset ID). 

1. The score is the label assigned by the user, and should be an integer that identifies their response. 

b) Full table format (suitable only when all classifiers respond to all data points). Each row corresponds to a subject, so that the row index is treated as the subjectID. Each column corresponds to a userID.

### Gold.csv (optional)

This file is required if you want to supply training data to IBCC (supervised or semi-supervised mode). It can also be supplied to evaluate the results of the IBCC classification using the built-in evaluation methods in this code package. The file can also be in one of two formats.

a) If the gold labels are present for every subject, we treat the row index as subjectID, and the gold file just contains a list of class labels.

b) Two column format:

1. subjectID 

1. gold label: an integer corresponding to the confirmed class label for the subject. The possible class labels should be contiguous integers starting from 0.

For a project where we are detecting rare objects or events, gold.csv should contain negative as well as positive confirmed examples. This allows proper evaluation of the false positive rate of IBCC or more effective training in supervised mode. 

## How To Run pyIBCC

### Prerequisites

The code was tested with python3. You need to install numpy and scipy. In Ubuntu, this is just `apt-get install python-numpy python-scipy`. 

### Running pyIBCC

Follow these steps. 

1. Save/check out the code, and make sure your input data is in the correct format. Example data files are provided, so test you can run the code with these first.

1. Set up the configuration file as described below so IBCC can find your input data.

1. In a command line, go to the directory where you downloaded the code, e.g. `cd ./pyIbcc/python`.

1. In the command line, call `python ibcc.py "<config_filename>"`. Replace <config_filename> with the path to your config file.

1. Look at the results in `./pyIbcc/python/output.csv` or wherever you configured the output.csv to be saved.

### Configuration

The configuration file "config/my_project.py" can be copied and modified to suit your current project. It's just a Python script that initialises some variables, which are described below. The last two of these -- the priors -- may sound confusing, but don't worry! If your volunteers' scores are direct predictions of the target classes, you should be able to stick with the defaults. E.g. if the volunteers say "there is an exoplanet in this candidate" and the target class is whether there really is an exoplanet in that image, you can use the default settings for the priors. If you want to adjust the priors to alter the IBCC results, consider the suggestions below and think about tweaking the parameters if IBCC doesn't work. 

1. `scores`: List of the scores that workers can provide. E.g. if user responses can be "3" or "4", then the line in the config file should read "scores = [3, 4]". This allows for projects where input.csv contains scores other than the gold labels.

1. `nClasses`: the number of target classes to predict.

1. `inputFile`: location of your input.csv.

1. `goldFile`: location of gold.csv.

1. `outputFile`: location of output.csv for target class predictions.

1. `confMatFile`: location to write out the confusion matrices for users. Leave it blank if you don't want to save this as it may be a large file.

1. `nu0`: prior for the class proportions; you may need to test a few different values to get good results. This is a vector where each element corresponds to a target class, e.g. one entry for "supernova" and one for "not supernova". The ratio of the values corresponds to your prior estimate of the ratio of the target classes. The magnitude corresponds to your confidence in your prior estimate of this ratio. If you are unsure what values to set for `nu0`, a good choice is to try values where the sum of elements of `nu0 = 100`.  E.g. if you are detecting a rare object such as a supernova, try `nu0=[99, 1]`. If the classes are likely to have equal proportions, you can try `nu0=[50,50]`. 

1. `alpha0`: prior for the volunteers' confusion matrices. You may also need to try different values here; methods for optimising this automatically have yet to be added to the code. This is a matrix of "pseudo-counts", meaning that each entry represents imaginary observations of a new volunteer's behaviour. A matrix element `alpha_{i,j}` is the pseudo-count for objects of true class `i` where the volunteer responds with the `j`th score. Setting the diagonal elements to larger values than the off-diagonals encodes the belief that volunteers are better than random. The magnitude of the counts controls the strength of the prior beliefs compared to what you learn about the volunteers from the data. From experience with Supernovae data, you should try setting the values so that the sum of each row is between `number of columns` and `number of columns * 2`. 

## Outputs from IBCC

The code will write the outputs to csv files.

1. `outputs.csv`: write out the predictions for the target class labels. Has the following columns: subjectID, p(t=0), p(t=1),... PyIBCC will write one column for each of the target classes, so that a binary problem will result in an output file with two columns. 

1. `confMat.csv`: the expected confusion matrices are output as a CSV file. Each row is a flattened confusion matrix for a single worker. Each entry is the likelihood of the classifier's responses given objects of each target class. The confusion matrix for each classifier is therefore flattened into a single row by laying each row of the matrix end-to-end. For example, if we have  a 2-by-2 confusion matrix, line `k` in confMat.csv shows the confusion matrix `\pi` for worker `k`. Column 1 in confMat.csv corresponds to `\pi_11`, column 2 is `\pi_12`, column 3 is `\pi_21`, column 4 is `\pi_22`.

# How the code works

This section will provide a short overview of how the code works to give you an idea of how to integrate or translate the existing code.

This page links to a Python implementation -- Matlab code is also available for tracking changing user behaviour (DynIBCC) by contacting edwin@robots.ox.ac.uk (needs more work to make it easy to use).
