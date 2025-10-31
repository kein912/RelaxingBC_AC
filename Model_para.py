import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

model = ["_DT","_RF", "_SVM", "_LR","_GP", "_KN", "_AD", "_GB", "_MLP", ]
# 
modelFunc = [DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression(penalty='l2'), GaussianProcessClassifier(), KNeighborsClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), MLPClassifier()]
# , SVC()
model_para_dic = {
                DecisionTreeClassifier(): 
                {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random'),
                 'min_samples_split': (2, 0.1, 0.5), 'min_samples_leaf': (1, 5),
                 'max_features': ('sqrt', 'log2', None), 
                 'random_state': [0,42]},

                #  DecisionTreeClassifier(): 
                # {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random'),\
                #  'min_samples_split': (0.1, 0.5), 'min_samples_leaf': (5, 10)},
                #  DecisionTreeClassifier(): 
                # {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random'),\
                #  'min_samples_split': (0.1, 0.5), 'min_samples_leaf': (5, 10)},

                RandomForestClassifier():
                {'n_estimators':range(10,100, 10),#[200, 500],
                 'max_features':['sqrt', 'log2'],
                 'max_depth': [3,4],
                 'criterion': ['gini','entropy'], 
                 'random_state': [0,42],
                },

                SVC():
                {'kernel': ['rbf','linear', 'poly'],
                'gamma': [1e-3, 1e-4, 'scale'],
                'probability':[True],
                'C': range(1,10), 
                'random_state': [0, 42],
                "class_weight":[ {0:2},  {0:3}, 'balanced'],
                },

                LogisticRegression(penalty='l2'):  # solver='liblinear' penalty='l2'
                {'C': np.logspace(-2, 2, 5),
                'solver': ('lbfgs', 'newton-cg', 'sag'),
                'max_iter': tuple(range(100, 1100, 200)), 
                'fit_intercept': (True, False), 
                'random_state':[0,42]},

                GaussianProcessClassifier():
                {
                    "kernel":[1.0 * RBF(1.0)],
                    "warm_start":[True, False],
                    "max_iter_predict":[100,150,200], 
                    'random_state':[0,42]
                },

                KNeighborsClassifier():
                {
                    "n_neighbors":range(2, 20),
                    "weights":["uniform", "distance"],
                    "algorithm":["ball_tree", "kd_tree", "brute"],
                },

                AdaBoostClassifier():
                {
                    "base_estimator":[DecisionTreeClassifier(), RandomForestClassifier(), GaussianProcessClassifier(), SVC()],
                    "n_estimators":[10, 20, 50,100],
                    "learning_rate":[0.5, 1.0],
                    "algorithm":["SAMME", "SAMME.R"], 
                    'random_state': [0,42],
                },

                GradientBoostingClassifier():
                {
                    "loss":["log_loss", "exponential"],#學長的code"loss":["deviance", "exponential"],
                    "learning_rate":[0.1, 0.2, 0.5],
                    "criterion":["friedman_mse", "squared_error"], #學長的code"criterion":["friedman_mse", "mse"], 
                    'random_state': [0,42],
                },

                MLPClassifier():
                {
                    'hidden_layer_sizes' :[(2, ), (3,), (4,)],
                    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                    'solver' : ['lbfgs', 'sgd', 'adam',],
                    'alpha' : [0.0001, 0.001,  0.01],
                    'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                    'max_iter' : [50, 70, 100],
                    'random_state': [0, 42],
                },

               
                #   KerasClassifier(build_fn=NN_seq_model_construct):
                # {'batch_size' : (2, 5),'epochs': (100,300)
                #   },

                #   # can only use for 3type 且分起來好像沒比較好
                #   KerasClassifier(build_fn=NN_multi_modal_construct):
                # {'batch_size' : (10, 20, 40, 60, 100),'epochs': (10, 50, 100)
                #   },
                # NuSVC():
                # {
                #     'cache_size' : [40],'degree':[1,3,5],'nu':[0.5],'gamma':['auto'],'kernel':['rbf'],'random_state': range(0, 50, 5)
                # }
}