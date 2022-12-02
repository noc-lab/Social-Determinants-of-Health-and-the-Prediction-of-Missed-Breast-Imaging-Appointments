
"""
***********************************************************************************************************************************
***********************************************************************************************************************************
                              Python implementation of predictive models introduced in the paper: 
                    "Social Determinants of Health and the Prediction of Missed Breast Imaging Appointments" 
                                                             By
     Shahabeddin Sotudian, Aaron Afran, Christina A. LeBedis, Anna F. Rives, Ioannis Ch. Paschalidis & Michael D. C. Fishman
***********************************************************************************************************************************
***********************************************************************************************************************************
"""  

from __future__ import division
import numpy as np
import cvxpy as cp
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from sklearn import preprocessing
from matplotlib import pyplot as plt


# ------------------------------------------------------------------------------
#  Data
# ------------------------------------------------------------------------------


# Load Processed Data  ----------------------
Final_Thrive_Data = pd.read_csv('./FinalData_Thrive2022.csv')

X = Final_Thrive_Data.drop(columns=['ORDER STATUS']) 
Y = pd.DataFrame( Final_Thrive_Data.loc[:]['ORDER STATUS'])


# Convert to dummy   ----------------------
cat_col=['PRIMARY RACE', 'LANGUAGE', 'HISPANIC INDICATOR','EDUCATION LEVEL', 'MARITAL STATUS',
         'WEEKDAY', 'TIME', 'MONTH',
         'PRIMARY INSURANCE','DEPARTMENT NAME','ORDER NAME']
                   

X =pd.get_dummies(X, prefix=cat_col, columns=cat_col,drop_first=True)

del cat_col


# Normalization  ----------------------
Set_Contious=[ 'AGE','Diff_schd_App', 'Total_app', 'Total_cancellations', 'Total_complete',
               'Days_since_last_app', 'Days_since_last_cancellation',
               'cancelation_last_appointment', 'Min_temperature',
               'precipIntensity', 'Median_household_income',
               'Distance']

X_Unnormalize=X
Part1= X.drop(columns=Set_Contious)
Part2= X[Set_Contious]

Normalization_scaler = preprocessing.MinMaxScaler()
Normalized_Part2=pd.DataFrame(   Normalization_scaler.fit_transform(Part2.values)    )
Normalized_Part2.columns=Part2.columns

X = pd.concat([Part1, Normalized_Part2], axis=1)
X=X.reset_index(drop = True)

del Part1,Part2,Normalized_Part2,Normalization_scaler

# -----------------------------------------------------------------------------
#     Feature Selection                                           
# -----------------------------------------------------------------------------
 
Y[Y==0] = -1   # Change the label
N_data = X.shape[0]

m = int(N_data* 0.7)
FS_X_Train = np.array( X[0:m][:]) 
FS_Y_Train = np.array( Y[0:m][:])
FS_X_Valid = np.array( X[m:N_data][:])
FS_Y_Valid = np.array(Y[m:N_data][:])

n = FS_X_Train.shape[1]
m = FS_X_Train.shape[0]-1    
    
np.random.seed(10)


# Form SVM with L1 regularization problem
beta = cp.Variable((n,1))
v = cp.Variable()
loss = cp.sum(cp.pos(1 - cp.multiply(FS_Y_Train, FS_X_Train @ beta - v)))
reg = cp.norm(beta, 1)
lambd = cp.Parameter(nonneg=True)
prob = cp.Problem(cp.Minimize(loss/m + lambd*reg))    # cp.Problem(cp.Minimize(loss/m + lambd*reg))


# Compute a trade-off curve and record train and test error.
lambda_vals = np.logspace(-50, -5, 45)
TRIALS = len(lambda_vals)
train_error = np.zeros(TRIALS)
test_error = np.zeros(TRIALS)
beta_vals = []
for i in range(TRIALS):
    print(i)
    lambd.value = lambda_vals[i]
    prob.solve(solver= 'GUROBI')
    train_error[i] = (FS_Y_Train != np.sign(FS_X_Train.dot(beta.value) - v.value)).sum()/m
    test_error[i] = (FS_Y_Valid != np.sign(FS_X_Valid.dot(beta.value) - v.value)).sum()/(N_data-m)
    beta_vals.append(beta.value)
    
    
# Plot the train and test error over the trade-off curve.
plt.plot(lambda_vals, train_error, label="Train error")
plt.plot(lambda_vals, test_error, label="Test error")
plt.xscale('log')
plt.legend(loc='upper left')
plt.xlabel(r"$\lambda$", fontsize=16)
plt.show()    

# Plot the regularization path for beta.
for i in range(n):
    plt.plot(lambda_vals, [wi[i,0] for wi in beta_vals])
plt.xlabel(r"$\lambda$", fontsize=16)
plt.xscale("log")





























