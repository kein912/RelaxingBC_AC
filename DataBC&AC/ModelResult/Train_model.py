import numpy as np
import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from grid_search_train import grid_search
from catboost import CatBoostClassifier

workDir = r'Relaxing/'
os.chdir(workDir)

def getData(dataset):
    # 'Fusion_AA'、'Fusion_NN'、'Fusion_ANpre'、'Fusion_ANpost'
    if dataset.split('_')[0].lower() == 'fusion' :
        return pd.read_csv("Pre/%s_feature.csv"%(dataset), index_col=0)
    else:
        return pd.read_csv('Pre/%s.csv'%(dataset), index_col=0)
    
def diffBetweenTrainingCount(data, g1, g2=None):
    if 'trainingCount' in g1:
        group1 = data[data['trainingCount'] == g1['trainingCount']]
    else:
        group1 = data

    group1 = group1[group1['addictLabel'] == g1['addictLabel']]
    # group1 = group1[group1['stateLabel'] == g1['stateLabel']].drop(['data_name', 'addictLabel', 'stateLabel', 'trainingCount'], axis=1).to_numpy()
    group1 = group1[group1['stateLabel'] == g1['stateLabel']].drop(['data_name', 'addictLabel', 'stateLabel'], axis=1).to_numpy()

    if g2:
        if 'trainingCount' in g2:
            group2 = data[data['trainingCount'] == g2['trainingCount']]
        else:
            group2 = data

        group2 = group2[group2['addictLabel'] == g2['addictLabel']]
        # group2 = group2[group2['stateLabel'] == g2['stateLabel']].drop(['data_name', 'addictLabel', 'stateLabel', 'trainingCount'], axis=1).to_numpy()
        group2 = group2[group2['stateLabel'] == g2['stateLabel']].drop(['data_name', 'addictLabel', 'stateLabel'], axis=1).to_numpy()
        data = group2 - group1
    else:
        data = group1

    return data

def createDataName(dataset, group1_1, group1_2, group2_1, group2_2, featureSelectModelName):
    dataName = ''
    dataName += dataset + '_'
    dataName +=  'A' if group1_1['addictLabel'] == 1 else 'N'

    if 'trainingCount' in group1_1:
        dataName += str(group1_1['trainingCount'])

    if group1_2 and 'trainingCount' in group1_2:
        dataName += str(group1_2['trainingCount'])

    if group2_1['addictLabel'] == 0:
        dataName += 'N'
    else:
        dataName += 'A'
        if 'trainingCount' in group2_1:
            dataName += str(group2_1['trainingCount'])

    if group2_2 and 'trainingCount' in group2_2:
        dataName += str(group2_2['trainingCount'])

    state_label_mapping = {0: 'pre', 1: 'vr', 2: 'post'}
    stage = state_label_mapping[group1_1['stateLabel']]
    stage2 = state_label_mapping[group2_1['stateLabel']]
    dataName += '_' + stage + '_' + stage2 + '_' + featureSelectModelName if featureSelectModelName != None else '_' + stage + '_' + stage2

    dataName += '.csv'
    return dataName


def trainModel(dataset, group1_1, group1_2, group2_1, group2_2,featureSelectModel=None,isleave_one=True):
    data = getData(dataset)
    data = data.drop_duplicates()

    x1 = diffBetweenTrainingCount(data, group1_1, group1_2)
    y1 = [0 for _ in range(0, len(x1))]

    x2 = diffBetweenTrainingCount(data, group2_1, group2_2)
    y2 = [1 for _ in range(0, len(x2))]

    X_train = np.concatenate([x1, x2])
    y_train = y1+y2

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    
    dataName = createDataName(dataset, group1_1, group1_2, group2_1, group2_2, type(featureSelectModel).__name__ if featureSelectModel != None else None)
    print(dataName)

    best_model = None
    best_accuracy = 0
    
    try:
        best_para = grid_search(X_train, y_train, leave_one=isleave_one,testSize=0.2,selectModel=featureSelectModel)#, title='HRV_prepost_DT', paraPath="Data/modelResult/")
        best_para.to_csv("ModelResult/Pre/%s"%(dataName))
    except ValueError as e:
        # 如果上述代碼塊產生ValueError，則會執行這個代碼塊
        print(dataName,"有問題無法執行")
        print("錯誤訊息",e)

    

    return dataName


#datasets = ['GSR','RES','HRV','EEG_bandMean','fusion']
# datasets = ['EEG'、'EEG_PCR'、'GSR'、'HRV'、'RES_dc'、'Fusion'、'Fusion_U-test_pp'、'Fusion_Ef_pp'、'Fusion_AA'、'Fusion_NN'、'Fusion_ANpre'、'Fusion_ANpost'、'Fusion_AA_Effect'、'Fusion_NN_Effect'、'Fusion_ANpre_Effect'、'Fusion_ANpost_Effect']   dc:drop column
datasets = ['Fusion_Ef_pp']
# selectModels = [None,LogisticRegression(),DecisionTreeClassifier(),LinearSVC(),CatBoostClassifier()]
selectModels = [None]


#A vs N pre Accuracy
for dataset in datasets:
    for selectModel in selectModels:
         # addictLabel 1A 0N stateLabel 0pre 2post
        group1_1 = {'addictLabel':1, 'stateLabel':2}
        group1_2 = None
        group2_1 = {'addictLabel':1, 'stateLabel':0}
        group2_2 = None
        trainModel(dataset, group1_1, group1_2, group2_1, group2_2, selectModel)