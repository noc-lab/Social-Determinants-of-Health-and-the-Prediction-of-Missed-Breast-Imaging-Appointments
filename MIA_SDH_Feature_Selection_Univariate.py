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
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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
#     Univariate Feature Selection  (UFS)                                          
# -----------------------------------------------------------------------------

Y[Y==0] = -1   # Change the label
N_data = X.shape[0]
m = int(N_data* 0.7)
FS_X_Train = ( X[0:m][:]) #31116 15%
FS_Y_Train = ( Y[0:m][:])
FS_X_Valid = ( X[m:N_data][:])
FS_Y_Valid = (Y[m:N_data][:])

n = FS_X_Train.shape[1]
m = FS_X_Train.shape[0]-1    
    
np.random.seed(10)


N_Features = 25
SELECTOR = SelectKBest(chi2, k=N_Features)
SELECTOR.fit_transform(FS_X_Train, FS_Y_Train)
cols = SELECTOR.get_support(indices=True)
features_df_new = FS_X_Train.iloc[:,cols]

UFS_Selected_Features = list(features_df_new.columns)


