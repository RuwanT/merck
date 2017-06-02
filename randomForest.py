import scipy
import math
import numpy as np
import pandas as pd
import plotly.plotly as py
import os.path
import sys

from time import time
from sklearn import preprocessing, metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

fname = str(raw_input('Please enter the input file name containing total dataset and descriptors (assumes csv file, column headings and first column are labels\n'))
if os.path.isfile(fname) :
    SubFeAll = pd.read_csv(fname, sep=",")
else:
    sys.exit("ERROR: input file does not exist")

#SubFeAll = pd.read_csv(fname, sep=",")
SubFeAll = SubFeAll.fillna(SubFeAll.mean()) # replace the NA values with the mean of the descriptor
header = SubFeAll.columns.values # Use the column headers as the descriptor labels
SubFeAll.head()

# Set the numpy global random number seed (similar effect to random_state) 
np.random.seed(1)  

# Random Forest results initialised
RFr2 = []
RFmse = []
RFrmse = []

# Predictions results initialised 
RFpredictions = []

metcount = 0

# Give the array from pandas to numpy
npArray = np.array(SubFeAll)
print header.shape
npheader = np.array(header[1:-1])
print("Array shape X = %d, Y = %d " % (npArray.shape))
datax, datay =  npArray.shape

# Print specific nparray values to check the data
print("The first element of the input data set, as a minial check please ensure this is as expected = %s" % npArray[0,0])

# Split the data into: names labels of the molecules ; y the True results ; X the descriptors for each data point
names = npArray[:,0]
X = npArray[:,2:-1].astype(float)

y = npArray[:,1] .astype(float)
X = preprocessing.scale(X)
print X.shape

# Open output files
train_name = "Training.csv"
test_name = "Predictions.csv"
fi_name = "Feature_importance.csv"

with open(train_name,'w') as ftrain, open(test_name,'w') as fpred, open(fi_name,'w') as ffeatimp:
    ftrain.write("This file contains the training information for the Random Forest models\n")
    ftrain.write("The code use a ten fold cross validation 90% training 10% test at each fold so ten training sets are used here,\n")
    ftrain.write("Interation %d ,\n" %(metcount+1))

    fpred.write("This file contains the prediction information for the Random Forest models\n")
    fpred.write("Predictions are made over a ten fold cross validation hence training on 90% test on 10%. The final prediction are return iteratively over this ten fold cros validation once,\n")
    fpred.write("optimised parameters are located via a grid search at each fold,\n")
    fpred.write("Interation %d ,\n" %(metcount+1))

    ffeatimp.write("This file contains the feature importance information for the Random Forest model,\n")
    ffeatimp.write("Interation %d ,\n" %(metcount+1))

    # Begin the K-fold cross validation over ten folds
    kf = KFold(datax, n_folds=10, shuffle=True, random_state=0)
    print "------------------- Begining Ten Fold Cross Validation -------------------"
    for train, test in kf:
        XTrain, XTest, yTrain, yTest = X[train], X[test], y[train], y[test]
        ytestdim = yTest.shape[0]
        print("The test set values are : ")
        i = 0
        if ytestdim%5 == 0:
            while i < ytestdim:
                print round(yTest[i],2),'\t', round(yTest[i+1],2),'\t', round(yTest[i+2],2),'\t', round(yTest[i+3],2),'\t', round(yTest[i+4],2)
                ftrain.write(str(round(yTest[i],2))+','+ str(round(yTest[i+1],2))+','+str(round(yTest[i+2],2))+','+str(round(yTest[i+3],2))+','+str(round(yTest[i+4],2))+',\n')
                i += 5
        elif ytestdim%4 == 0:
            while i < ytestdim:
                print round(yTest[i],2),'\t', round(yTest[i+1],2),'\t', round(yTest[i+2],2),'\t', round(yTest[i+3],2)
                ftrain.write(str(round(yTest[i],2))+','+str(round(yTest[i+1],2))+','+str(round(yTest[i+2],2))+','+str(round(yTest[i+3],2))+',\n')
                i += 4
        elif ytestdim%3 == 0 :
            while i < ytestdim :
                print round(yTest[i],2),'\t', round(yTest[i+1],2),'\t', round(yTest[i+2],2)
                ftrain.write(str(round(yTest[i],2))+','+str(round(yTest[i+1],2))+','+str(round(yTest[i+2],2))+',\n')
                i += 3
        elif ytestdim%2 == 0 :
            while i < ytestdim :
                print round(yTest[i],2), '\t', round(yTest[i+1],2)
                ftrain.write(str(round(yTest[i],2))+','+str(round(yTest[i+1],2))+',\n')
                i += 2
        else :
            while i< ytestdim :
                print round(yTest[i],2)
                ftrain.write(str(round(yTest[i],2))+',\n')
                i += 1        

                print "\n"
            # random forest grid search parameters
            print "------------------- Begining Random Forest Grid Search -------------------"
            rfparamgrid = {"n_estimators": [10], "max_features": ["auto", "sqrt", "log2"], "max_depth": [5,7]}
            rf = RandomForestRegressor(random_state=0,n_jobs=2)
            RfGridSearch = GridSearchCV(rf,param_grid=rfparamgrid,scoring='mean_squared_error',cv=10)
            start = time()
            RfGridSearch.fit(XTrain,yTrain)

            # Get best random forest parameters
            print("GridSearchCV took %.2f seconds for %d candidate parameter settings" %(time() - start,len(RfGridSearch.grid_scores_)))
            RFtime = time() - start,len(RfGridSearch.grid_scores_)
            #print(RfGridSearch.grid_scores_)  # Diagnos
            print("n_estimators = %d " % RfGridSearch.best_params_['n_estimators'])
            ne = RfGridSearch.best_params_['n_estimators']
            print("max_features = %s " % RfGridSearch.best_params_['max_features'])
            mf = RfGridSearch.best_params_['max_features']
            print("max_depth = %d " % RfGridSearch.best_params_['max_depth'])
            md = RfGridSearch.best_params_['max_depth']

            ftrain.write("Random Forest")
            ftrain.write("RF search time, %s ,\n" % (str(RFtime)))
            ftrain.write("Number of Trees, %s ,\n" % str(ne))
            ftrain.write("Number of feature at split, %s ,\n" % str(mf))
            ftrain.write("Max depth of tree, %s ,\n" % str(md))

            # Train random forest and predict with optimised parameters
            print("\n\n------------------- Starting opitimised RF training -------------------")
            optRF = RandomForestRegressor(n_estimators = ne, max_features = mf, max_depth = md, random_state=0)
            optRF.fit(XTrain, yTrain)       # Train the model
            RFfeatimp = optRF.feature_importances_
            indices = np.argsort(RFfeatimp)[::-1]
            print("Training R2 = %5.2f" % optRF.score(XTrain,yTrain))
            print("Starting optimised RF prediction")
            RFpreds = optRF.predict(XTest)
            print("The predicted values now follow :")
            RFpredsdim = RFpreds.shape[0]
            i = 0
            if RFpredsdim%5 == 0:
                while i < RFpredsdim:
                    print round(RFpreds[i],2),'\t', round(RFpreds[i+1],2),'\t', round(RFpreds[i+2],2),'\t', round(RFpreds[i+3],2),'\t', round(RFpreds[i+4],2)
                    i += 5
            elif RFpredsdim%4 == 0:
                while i < RFpredsdim:
                    print round(RFpreds[i],2),'\t', round(RFpreds[i+1],2),'\t', round(RFpreds[i+2],2),'\t', round(RFpreds[i+3],2)
                    i += 4
            elif RFpredsdim%3 == 0 :
                while i < RFpredsdim :
                    print round(RFpreds[i],2),'\t', round(RFpreds[i+1],2),'\t', round(RFpreds[i+2],2)
                    i += 3
            elif RFpredsdim%2 == 0 :
                while i < RFpredsdim :
                    print round(RFpreds[i],2), '\t', round(RFpreds[i+1],2)
                    i += 2
            else :
                while i< RFpredsdim :
                    print round(RFpreds[i],2)
                    i += 1
                    print "\n"
                    RFr2.append(optRF.score(XTest, yTest))
                    RFmse.append( metrics.mean_squared_error(yTest,RFpreds))
                    RFrmse.append(math.sqrt(RFmse[metcount]))
                    print ("Random Forest prediction statistics for fold %d are; MSE = %5.2f RMSE = %5.2f R2 = %5.2f\n\n" % (metcount+1, RFmse[metcount], RFrmse[metcount],RFr2[metcount]))

                    ftrain.write("Random Forest prediction statistics for fold %d are, MSE =, %5.2f, RMSE =, %5.2f, R2 =, %5.2f,\n\n" % (metcount+1, RFmse[metcount], RFrmse[metcount],RFr2[metcount]))



                    ffeatimp.write("Feature importance rankings from random forest,\n")
                    for i in range(RFfeatimp.shape[0]) :
                        ffeatimp.write("%d. , feature %d , %s,  (%f),\n" % (i + 1, indices[i], npheader[indices[i]], RFfeatimp[indices[i]]))


            # Store prediction in original order of data (itest) whilst following through the current test set order (j)
            metcount += 1

            ftrain.write("Fold %d, \n" %(metcount))
