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

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import statsmodels.stats.proportion as smprop
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from sklearn import preprocessing
from scipy import stats
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import random
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,f1_score, roc_curve, auc     
from sklearn.model_selection import  StratifiedKFold,cross_val_score 
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb 
from sklearn.metrics import confusion_matrix
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


# -----------------------------------------------------------------------------
#   FUNCTIONS
# ------------------------------------------------------------------------------

def stat_test(df, y):
    name = pd.DataFrame(df.columns,columns=['Variable'])
    df0=df[y==0]
    df1=df[y==1]
    pvalue=[]
    y_corr=[]
    for col in df.columns:
        if df[col].nunique()==2:
            zstat, pval = smprop.proportions_ztest([df0[col].sum() , df1[col].sum()], [len(df0[col]) , len(df1[col])], value=None, alternative='two-sided', prop_var=False)
            pvalue.append(pval)
        else:
            pvalue.append(stats.ks_2samp(df0[col], df1[col]).pvalue)
            
        y_corr.append(df[col].corr(y))
    name['All_mean']=df.mean().values
    name['y1_mean']=df1.mean().values
    name['y0_mean']=df0.mean().values
    name['All_std']=df.std().values
    name['y1_std']=df1.std().values
    name['y0_std']=df0.std().values
    name['p-value']=pvalue
    name['y_corr']=y_corr
    return name.sort_values(by=['p-value'])



def high_corr(df, thres=0.8):
    corr_matrix_raw = df.corr()
    corr_matrix = corr_matrix_raw.abs()
    high_corr_var_=np.where(corr_matrix>thres)
    high_corr_var=[(corr_matrix.index[x],corr_matrix.columns[y], corr_matrix_raw.iloc[x,y]) for x,y in zip(*high_corr_var_) if x!=y and x<y]
    return high_corr_var


def df_fillna(df):
    df_nullsum=df.isnull().sum()
    for col in df_nullsum[df_nullsum>0].index:
        df[col+'_isnull']=df[col].isnull()
        df[col]=df[col].fillna(df[col].median())    
    return df


def df_drop(df_new, drop_cols):
    return df_new.drop(df_new.columns[df_new.columns.isin(drop_cols)], axis=1)
  

def clf_F1(best_C_grid, best_F1, best_F1std, classifier, X_train, y_train,C_grid,nFolds, silent=True,seed=2020):
    results= cross_val_score(classifier, X_train, y_train, cv=StratifiedKFold(n_splits=nFolds,shuffle=True,random_state=seed), n_jobs=-1,scoring='f1')
    F1, F1std = results.mean(), results.std()
    if silent==False:
        print(C_grid,F1, F1std)        
    if F1>best_F1:
        best_C_grid=C_grid
        best_F1, best_F1std=F1, F1std
    return best_C_grid, best_F1, best_F1std

def my_RFE(df_new, col_y='Hospitalization', my_range=range(5,60,2), my_penalty='l1', my_C = 0.01, cvFolds=5,step=1):
    F1_all_rfe=[]
    Xraw=df_new.drop(col_y, axis=1).values
    y= df_new[col_y].values
    names=df_new.drop(col_y, axis=1).columns
    for n_select in my_range:
        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(Xraw)
        clf=LogisticRegression(C=my_C,penalty=my_penalty,class_weight= 'balanced',solver='liblinear')
        rfe = RFE(clf, n_select, step=step)
        rfe.fit(X, y.ravel())
        X = df_new.drop(col_y, axis=1).drop(names[rfe.ranking_>1], axis=1).values  
        X = scaler.fit_transform(X)
        best_F1, best_F1std=0.1, 0
        best_C_grid=0
        for C_grid in [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]:
            clf=LogisticRegression(C=C_grid,class_weight= 'balanced',solver='liblinear')
            best_C_grid, best_F1, best_F1std=clf_F1(best_C_grid, best_F1, best_F1std,clf,X, y,C_grid,cvFolds)
        F1_all_rfe.append((n_select, best_F1, best_F1std))
    F1_all_rfe=pd.DataFrame(F1_all_rfe, index=my_range,columns=['n_select',"best_F1","best_F1std"])
    F1_all_rfe['F1_']= F1_all_rfe['best_F1']-F1_all_rfe['best_F1std']
    X = scaler.fit_transform(Xraw)
    clf=LogisticRegression(C=my_C,penalty=my_penalty,class_weight= 'balanced',solver='liblinear')
    rfe = RFE(clf, F1_all_rfe.loc[F1_all_rfe['F1_'].idxmax(),'n_select'], step=step)
    rfe.fit(X, y.ravel())
    id_keep_1st= names[rfe.ranking_==1].values
    return id_keep_1st, F1_all_rfe


def my_train(X_train, y_train, model='LR', penalty='l1', cv=5, scoring='f1', class_weight= 'balanced',seed=2020):    
    if model=='SVM':
        svc=LinearSVC(penalty=penalty, class_weight= class_weight, dual=False, max_iter=5000)
        parameters = {'C':[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]}  
        gsearch = GridSearchCV(svc, parameters, cv=cv, scoring=scoring) 
    elif model=='LGB':        
        param_grid = {
            'num_leaves': range(6,15,2),
            'n_estimators': [100,500,1000],
            'colsample_bytree': [0.3, 0.7, 0.9]
            }
        lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', learning_rate=0.1, random_state=seed)
        gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=cv,n_jobs=-1, scoring=scoring)
    elif model=='RF': 
        rfc=RandomForestClassifier(random_state=seed, class_weight= class_weight, n_jobs=-1)
        param_grid = { 
            'max_features':[0.4, 0.5, 0.6, 0.7, 0.8],
            'n_estimators': [50,100,500,1000,1500],
            'max_depth' : [2,5,10, 25],
            'min_samples_leaf' : [2, 5, 10],
            'min_samples_split' : [2, 5, 10], 
        }
        
        gsearch = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=cv, scoring=scoring)    
     
    else:
        LR = LogisticRegression(penalty=penalty, class_weight= class_weight,solver='liblinear', random_state=seed)
        parameters = {'C':[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000] } 
        gsearch = GridSearchCV(LR, parameters, cv=cv, scoring=scoring) 
    gsearch.fit(X_train, y_train)
    clf=gsearch.best_estimator_
    if model=='LGB' or model=='RF': 
        print('Best parameters found by grid search are:', gsearch.best_params_)
    print('train set accuracy:', gsearch.best_score_)
    return clf



def cal_f1_scores(thresholds, ytest, ytest_pred_score):
  thresholds = sorted(thresholds, reverse = True)
  metrics_all = []
  for thresh in thresholds:
    ytest_pred = np.array((ytest_pred_score > thresh))
    metrics_all.append(( thresh, f1_score(ytest, ytest_pred), f1_score(ytest, ytest_pred, average='micro'), f1_score(ytest, ytest_pred, average='macro'),f1_score(ytest, ytest_pred, average='weighted')))

  metrics_df = pd.DataFrame(metrics_all, columns=['thresh', 'F1-score', 'micro F1-score', 'macro F1-score','weighted F1-score'])
  
  return metrics_df



def my_test(xtest, ytest, clf, target_names, report=False, model='LR'): 
    ytest_pred=clf.predict(xtest)
    if model=='SVM': 
        ytest_pred_score=clf.decision_function(xtest)
    else:
        ytest_pred_score=clf.predict_proba(xtest)[:,1]
      
    fpr, tpr, thresholds = roc_curve(ytest, ytest_pred_score) 
    metrics_df = cal_f1_scores(thresholds, ytest, ytest_pred_score)
    metrics_df= metrics_df.sort_values(by = 'weighted F1-score', ascending = False)
    F1_values = metrics_df[['thresh', 'F1-score','micro F1-score','macro F1-score', 'weighted F1-score']].head(1).values

    cm2 = ConfusionMatrix(actual_vector=ytest, predict_vector=ytest_pred) 
    print('PPV_Macro :    ',cm2.PPV_Macro)
    print('PPV_Micro :    ',cm2.PPV_Micro )
    print('NPV :    ', cm2.NPV )
    print('PPV :    ', cm2.PPV )
  
    if report:
        print(classification_report(ytest, ytest_pred, target_names=target_names,digits=4))
    return (F1_values[0][1], auc(fpr, tpr),  F1_values[0][2], F1_values[0][3],F1_values[0][4])



def tr_predict(df_new, col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring='f1', test_size=0.2,report=False, RFE=False):
    scaler = preprocessing.MinMaxScaler()          
    metrics_all=[]
    my_seeds=range(2021, 2026)   
    for seed in my_seeds:
        # Train-Test Split last appt.
        Patients_IDs = set(df_new.ID.unique())
        random.seed(seed)
        Train_Patients = random.sample(Patients_IDs, round(len(Patients_IDs)*0.8))
        Test_Patients = Patients_IDs - set(Train_Patients)
        Train_List = []
        for PID in Train_Patients:
            A1 = df_new[df_new['ID']==PID]
            A2 = list(A1.loc[A1.index[len(A1)-1]][:].values)  #  last appoitment
            Train_List.append(A2)
            
        Training_Data = pd.DataFrame(Train_List, columns=list(df_new.columns))
        X_train = Training_Data.drop([col_y,'ID'], axis=1).values
        name_cols = Training_Data.drop([col_y,'ID'], axis=1).columns.values 
        X_train = scaler.fit_transform(X_train)
        y_train = Training_Data[col_y].values
        y_train = y_train.astype(np.int64)

        test_all_app = 0
        if test_all_app:
            Testing_Data = df_new[df_new['ID'].isin(Test_Patients)]
            xtest = Testing_Data.drop([col_y,'ID'], axis=1).values
            name_cols = Testing_Data.drop([col_y,'ID'], axis=1).columns.values 
            xtest = scaler.fit_transform(xtest)
            ytest = Testing_Data[col_y].values
            ytest = ytest.astype(np.int64)            
        else:
            Test_List = []    
            for PID in Test_Patients:
                A1 = df_new[df_new['ID']==PID]
                A2 = list(A1.loc[A1.index[len(A1)-1]][:].values)  #  last appoitment
                Test_List.append(A2)
            Testing_Data = pd.DataFrame(Test_List, columns=list(df_new.columns))
            xtest = Testing_Data.drop([col_y,'ID'], axis=1).values
            name_cols = Testing_Data.drop([col_y,'ID'], axis=1).columns.values 
            xtest = scaler.fit_transform(xtest)
            ytest = Testing_Data[col_y].values
            ytest = ytest.astype(np.int64)

        if RFE:
            df_train=pd.DataFrame(X_train, columns=name_cols )
            df_train[col_y]=y_train
            id_keep_1st, F1_all_rfe=my_RFE(df_train, col_y=col_y, cvFolds=cv_folds)
            print(F1_all_rfe)
            X_train=df_train[id_keep_1st]
            df_test=pd.DataFrame(xtest, columns=name_cols )
            xtest=df_test[id_keep_1st]
            name_cols=id_keep_1st
        clf = my_train(X_train, y_train, model=model, penalty=penalty, cv=cv_folds, scoring=scoring, class_weight= 'balanced',seed=seed)    
        metrics_all.append(my_test(xtest, ytest, clf, target_names, report=report, model=model))
    metrics_df = pd.DataFrame(metrics_all, columns=['F1-score', 'AUC', 'micro F1-score', 'macro F1-score','weighted F1-score'])

    if model=='LGB' or model=='RF': 
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.feature_importances_,2))),columns=['Variable','coef_'])
    else:
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.coef_[0],2))),columns=['Variable','coef_'])
    df_coef_['coef_abs']=df_coef_['coef_'].abs()
    return df_coef_.sort_values('coef_abs', ascending=False)[['Variable','coef_']], metrics_df



# -----------------------------------------------------------------------------
#     Model Training                                            
# -----------------------------------------------------------------------------

# Selected Feature selection

Selected_Features= ['ID',
                    'LANGUAGE_Spanish',
                    'Days_since_last_cancellation',
                    'MONTH_Spring',
                    'PRIMARY INSURANCE_PRIVATE',
                    'DEPARTMENT NAME_CHC',
                    'DEPARTMENT NAME_PrimaryCare',
                    'MONTH_Winter',
                    'ORDER NAME_Order_screening',
                    'cancelation_last_appointment',
                    'MARITAL STATUS_Married',
                    'TIME_After_16',
                    'Total_complete',
                    'DEPARTMENT NAME_Radiology',
                    'TIME_Before_8',
                    
                    'Min_temperature',
                    'Median_household_income',
                    'Distance',
                    
                    'Housing',
                    'Transportation',
                    'Utilities',
                    
                    ]

Manually_Add_SDHs = ['Food', 'Medications','Caregiving', 'Employment',  'Educ']


Final_Features =['ORDER STATUS'] + Selected_Features + Manually_Add_SDHs 

Data_X_Y =  pd.concat([Y, X], axis=1)
Data_X_Y=Data_X_Y[Final_Features]
Data_X_Y=Data_X_Y.reset_index(drop=True)
del Selected_Features,Manually_Add_SDHs,Y

# Parameters
Model_Names = ['LR','SVM','RF','LGB'] 
Penalty_Names =['l2','l2',' ',' ']  

print("***************************************************************************")
print("                 Performance evaluation - Train-Test                       ")
print("***************************************************************************",'\n')
print("Number of samples: ",Data_X_Y.shape[0],'\n') 
print("***************************************************************************",'\n')

for E in range(len(Model_Names)):
    print('\n' ,'*****      Alg. Name: ', (Model_Names[E]+ Penalty_Names[E]), '######=====-----------------   ','\n')
    df_coef_,metrics_df = tr_predict(Data_X_Y, col_y='ORDER STATUS',target_names = ['0', '1'], model= Model_Names[E] ,penalty= Penalty_Names[E] ,cv_folds=5,scoring='roc_auc', test_size=0.2,report=False)
    print('Final Result ===-----------------   ','\n', metrics_df[['AUC', 'micro F1-score','weighted F1-score','F1-score','macro F1-score']].describe().T[['mean','std']].stack().to_frame().T,'\n')
    # Importing libraries for dataframe creation 
    import seaborn as sns
    x_labels = [df_coef_['Variable'][i] for i in range(len(df_coef_['Variable']))]
    plt.figure(figsize=(25, 15)) 
    plots = sns.barplot(x="Variable", y= "coef_", data=df_coef_,label = 'Variable', palette="YlOrBr", edgecolor = 'k') 
    for bar in plots.patches: 
      plots.annotate(format(bar.get_height(), '.2f'),  
                       (bar.get_x() + bar.get_width() / 2,  
                        bar.get_height()), ha='center', va='center', 
                       size=15, xytext=(0, 8), 
                       textcoords='offset points') 

    plt.xlabel("Features", size=25, fontweight='bold') 
    plt.ylabel("Feature Importance", size=25, fontweight='bold') 
    plt.title(Model_Names[E] + ' Feature Importance', size=30, fontweight='bold') 
    plt.xticks(rotation = 90, fontsize=18)
    plt.yticks( fontsize=18)
    plt.show() 

del df_coef_,metrics_df,E,Model_Names,Penalty_Names,x_labels,plots,bar




print("***************************************************************************")
print("                 Marginal Effects Tables                       ")
print("***************************************************************************",'\n')
scaler = preprocessing.MinMaxScaler()          
my_seeds=range(2022, 2023)  
# Train-Test Split last appt.
Patients_IDs = set(Data_X_Y.ID.unique())
random.seed(my_seeds)
Train_Patients = random.sample(Patients_IDs, round(len(Patients_IDs)*0.9999))
Train_List = []
for PID in Train_Patients:
    A1 = Data_X_Y[Data_X_Y['ID']==PID]
    A2 = list(A1.loc[A1.index[len(A1)-1]][:].values)  #  last appoitment
    Train_List.append(A2)
    
Data_X_Y_LatApp = pd.DataFrame(Train_List, columns=list(Data_X_Y.columns))
Data_X_Y_LatApp = Data_X_Y_LatApp.drop(columns=['ID'])
    
    
# Logit model Normalized  
X_train = Data_X_Y_LatApp.drop(['ORDER STATUS'], axis=1).values  
X_train = scaler.fit_transform(X_train)        
name_cols = Data_X_Y_LatApp.drop(['ORDER STATUS'], axis=1).columns.values 
        
y_train = Data_X_Y_LatApp['ORDER STATUS'].values
y_train = y_train.astype(np.int64)

X_train = pd.DataFrame(X_train, columns=list(name_cols))
y_train = pd.DataFrame(y_train, columns=list(['ORDER STATUS']))

clf =  Logit( y_train  ,  sm.add_constant(X_train) ) 
clf_res = clf.fit(method='newton',maxiter=1000)

Coef_table_Normalized = pd.concat([clf_res.params, clf_res.conf_int()], axis=1)
Coef_table_Normalized.columns = ['Coefs_Normalized', 'Coef_2.5%_Normalized', 'Coef_97.5%_Normalized']



# Logit model UnNormalized      
logit = Logit( Data_X_Y_LatApp['ORDER STATUS']  ,  sm.add_constant(Data_X_Y_LatApp.drop(columns=['ORDER STATUS']) ) )
res = logit.fit(method='newton',maxiter=1000)
print(res.summary())
# marginal effects at the mean                                                        
Marginal_effect = res.get_margeff(at='overall',method="dydx",dummy= True , count= True ) 
print(Marginal_effect.summary())
print("-------------------------------------------------------------------------")


#  the confidence interval of each coeffecient
Coef_table = pd.concat([res.params, res.conf_int()], axis=1)
Coef_table.columns = ['Coefs', 'Coef_2.5%', 'Coef_97.5%']
Constant_Coef = Coef_table.loc['const']
Coef_table = Coef_table.drop(labels =  'const')
Coef_table['Variable'] = list(Coef_table.index)

# Marginal_effect
Coef_table['Marginal_effect'] = Marginal_effect.margeff
Coef_table['P_Values_Stat_Model'] =Marginal_effect.pvalues

# Some stats
Find_y0Mean_Y1_mean=stat_test(   Data_X_Y_LatApp.drop(columns=['ORDER STATUS'])  , Data_X_Y_LatApp['ORDER STATUS'])

# odds ratios and 95% CI
Oods_Ratio_table = pd.concat([res.params, res.conf_int()], axis=1)
Oods_Ratio_table.columns = ['Odds_Ratio', 'Odds_Ratio 2.5%', 'Odds_Ratio 97.5%']
Oods_Ratio_table = Oods_Ratio_table.drop(labels =  'const')
Oods_Ratio_table=np.exp(Oods_Ratio_table)

AAA_ss = Coef_table['Variable']
Coef_table = Coef_table.drop(columns =['Variable'])

Table_Coef_Odds_CIs=  pd.concat([AAA_ss, Coef_table, Oods_Ratio_table], axis=1)

del Coef_table,Oods_Ratio_table,AAA_ss

FINAl_TABLE=Table_Coef_Odds_CIs.merge(Find_y0Mean_Y1_mean, on='Variable')[['Variable',
                                                                             'Coefs', 'Coef_2.5%', 'Coef_97.5%',
                                                                             'Marginal_effect',
                                                                             'Odds_Ratio', 'Odds_Ratio 2.5%', 'Odds_Ratio 97.5%',
                                                                             'All_mean', 'y1_mean', 'y0_mean',
                                                                             'All_std', 'y1_std','y0_std', 'y_corr', 'p-value','P_Values_Stat_Model']]



FINAl_TABLE = FINAl_TABLE.append({  'Variable':'Constant_Coef', 
                                    'Coefs':Constant_Coef['Coefs'], 
                                    'Coef_2.5%':Constant_Coef['Coef_2.5%'], 
                                    'Coef_97.5%':Constant_Coef['Coef_97.5%'], 
                                    'Marginal_effect':0, 
                                    'Odds_Ratio':0, 
                                    'Odds_Ratio 2.5%':0, 
                                    'Odds_Ratio 97.5%':0, 
                                    'All_mean':0, 
                                    'y1_mean':0, 
                                    'y0_mean':0, 
                                    'All_std':0, 
                                    'y1_std':0, 
                                    'y0_std':0, 
                                    'y_corr':0, 
                                    'p-value':0,
                                    'P_Values_Stat_Model':0
                                     } , ignore_index=True)



Is_Significant = []
for i in range(len(FINAl_TABLE['P_Values_Stat_Model'])):
    if FINAl_TABLE['Variable'][i] == 'Constant_Coef':
        Is_Significant.append('NA')
    else:
        if FINAl_TABLE['P_Values_Stat_Model'][i] <= (0.05): # (0.05/(len(FINAl_TABLE)-1))
            Is_Significant.append('Significant')
        else:
            Is_Significant.append('Not Significant')

FINAl_TABLE['Significance_Bonferroni'] = Is_Significant

del Find_y0Mean_Y1_mean,  Table_Coef_Odds_CIs,Constant_Coef,i, Is_Significant,res,logit,Marginal_effect







