import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc

from Plot_learning_curve import plot_learning_curve
from Model_para import model
from Model_para import model_para_dic
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier


def specificity_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    # 'tn': cm[0, 0], 'fp': cm[0, 1],
    # 'fn': cm[1, 0], 'tp': cm[1, 1]
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])


def cal_f1_score(pred_array):
    # Ensure matrix is 2x2 by filling missing rows/columns with zeros
    if pred_array.shape[0] < 2:
        pred_array = np.pad(pred_array, ((0, 2 - pred_array.shape[0]), (0, 0)), mode='constant')
    if pred_array.shape[1] < 2:
        pred_array = np.pad(pred_array, ((0, 0), (0, 2 - pred_array.shape[1])), mode='constant')

    TP = pred_array[1, 1]
    TN = pred_array[0, 0]
    FN = pred_array[1, 0]
    FP = pred_array[0, 1]

    result = {}
    result['Accuracy'] = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    result['Precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0
    result['Recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0
    result['F1_Score'] = 2 * (result['Recall'] * result['Precision']) / (result['Recall'] + result['Precision']) if (result['Recall'] + result['Precision']) > 0 else 0

    return result


def save_model_para(gs, index):
    bestPara = {
        'model': model[index],
        'best_estimator': gs.best_estimator_,
        'best_para': gs.best_params_,
        'mean_test_accuracy': gs.cv_results_['mean_test_accuracy'][gs.best_index_],
        'std_test_accuracy': gs.cv_results_['std_test_accuracy'][gs.best_index_],
        'mean_test_specificity': gs.cv_results_['mean_test_specificity'][gs.best_index_],
        'std_test_specificity': gs.cv_results_['std_test_specificity'][gs.best_index_],
        'mean_test_sensitivity': gs.cv_results_['mean_test_sensitivity'][gs.best_index_],
        'std_test_sensitivity': gs.cv_results_['std_test_sensitivity'][gs.best_index_],
        'mean_test_AUC': gs.cv_results_['mean_test_AUC'][gs.best_index_]}

    return bestPara


def grid_search(
    X,
    y,
    X_test=[],
    y_test=[],
    selectModel=None,
    title="",
    paraPath="",
    imgPath="",
    testSize=0,
    leave_one=False,
):
    X = pd.DataFrame(X)
    X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(
        X), columns=X.columns, index=X.index)

    y = pd.DataFrame(y)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    if selectModel != None:
        selector = SelectFromModel(estimator=selectModel).fit(X, y)
        # feature_names = X.columns  # 保存列名
        feature_names = X.columns.tolist()  # 獲取特徵名稱列表
        X = selector.transform(X)
        # feature_names_series = pd.Series(feature_names)
        # selected_features = feature_names[selector.get_support()]
        # selected_features = feature_names_series[selector.get_support()].tolist()
        support = selector.get_support()  # 獲取布爾索引
        selected_features_indices = np.where(support)[0]  # 將布爾索引轉換為整數索引
        selected_features = [feature_names[i] for i in selected_features_indices]  # 提取選中的特徵名稱
        print("selected_features : ", selected_features)

    if leave_one == True:
        print('test with leave one out')
        X_train, X_test, y_train, y_test = X, X, y, y
    elif not X_test.empty and not y_test.empty:
        print('test from outside')
        X_train, X_test, y_train, y_test = X, X_test, y, y_test
    elif (testSize != 0):
        print('test from inside')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize,
                                                            random_state=0,
                                                            stratify=y)
    else:
        print('no test')
        X_train, X_test, y_train, y_test = X, X, y, y

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
    scoring = {'accuracy': make_scorer(
        accuracy_score), 'specificity': specificity_matrix_scorer, 'sensitivity': 'recall', 'AUC': 'roc_auc'}

    bestPara = pd.DataFrame()
    for i, m in enumerate(model_para_dic):
        if m == CatBoostClassifier:
            mod = m()
        else:
            mod = m
        # if i == 6:
        #     continue
        # grid_search
        # set = rskf.split(X_train, y_train)
        gs = GridSearchCV(
            estimator=mod,
            param_grid=model_para_dic[m],
            scoring=scoring,
            refit='AUC',
            # cv=set,
            n_jobs=4,
            verbose=0
        )
        gs.fit(X_train, y_train)

        # draw the learning curve
        title_ = title + model[i]

        clf = m
        clf.set_params(**gs.best_params_)

        plot_learning_curve(
            clf,
            title_,
            X_train,
            y_train,
        )

        if leave_one:
            y_pred = leave_one_out(X_train, y_train, m, gs.best_params_)
        else:
            y_pred = gs.best_estimator_.predict(X_test)

        cf = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        plt.title(model[i])
        sns.heatmap(cf, annot=True)

        if imgPath != "" and title != "":
            plt.savefig(imgPath + title + model[i] + ".png")

        plt.show()
        # plt.clf()

        draw_ROC_curve(y_test, y_pred)

        # save parameter of gs result
        para = save_model_para(gs, i)
        para.update(cal_f1_score(cf))
        para['kappa_score'] = sklearn.metrics.cohen_kappa_score(y_test, y_pred)
        # bestPara = bestPara.append(
        #     para,
        #     ignore_index=True
        # )
        para_df = pd.DataFrame([para])
        bestPara = pd.concat([bestPara, para_df], ignore_index=True, axis=0)

    if paraPath != "" and title != "":
        bestPara.to_csv(paraPath+title+".csv")
        print(paraPath+title+".csv")
    return bestPara


def leave_one_out(X, y, m, para):
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.DataFrame(y).reset_index(drop=True)

    y_pred = []
    for i, x in X.iterrows():
        X_train = X.drop(i)
        y_train = y.drop(i)
        y_train = np.ravel(y_train)

        x = x.to_frame().T
        clf = m
        clf.set_params(**para)

        clf.fit(X_train, y_train)
        pred = clf.predict(x)
        y_pred.append(pred[0])

    return y_pred


def draw_ROC_curve(y, y_pred, color='orange'):
    fpr, tpr, threshold = roc_curve(y, y_pred)

    auc1 = auc(fpr, tpr)
    # Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc1)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
