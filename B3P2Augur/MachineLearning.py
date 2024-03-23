import pandas
import numpy
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import lightgbm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class MachineLearning(object):
    def __init__(self, file):
        self.file = file
        self.training_dataframe = None
        self.training_datalabel = None
        self.training_score = None
        self.testing_dataframe = None
        self.testing_datalabel = None
        self.testing_score = None
        self.best_model = None
        self.best_n_trees = 0
        self.metrics = None
        self.aucData = None
        self.prcData = None
        self.meanAucData = None
        self.meanPrcData = None
        self.indepAucData = None
        self.indepPrcData = None
        self.algorithm = None
        self.error_msg = None
        self.boxplot_data = None
        self.message = None
        self.task = None

    def data_import(self, file):
        f = pandas.read_csv(file, sep=',', header=None)
        self.training_dataframe = f.iloc[:, 1:]
        self.row = self.training_dataframe.index.size
        self.column = self.training_dataframe.columns.size
        self.training_dataframe.index = ['Sample_%s' % i for i in range(self.row)]
        self.training_dataframe.columns = ['F_%s' % i for i in range(self.column)]
        self.training_datalabel = numpy.array(f.iloc[:, 0]).astype(int)


class RF(MachineLearning):
    def __init__(self, file):
        super(RF, self).__init__(file)

    def RF(self):
        fold = 5
        tree_range = (50, 300, 10)
        categories = sorted(set(self.training_datalabel))
        X, y = self.training_dataframe.values, self.training_datalabel
        best_n_trees = tree_range[0]
        best_auc = 0
        best_accuracy = 0
        best_model = []
        best_training_score = None

        for tree in range(tree_range[0], tree_range[1] + 1, tree_range[2]):
            training_score = numpy.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                rfc_model = RandomForestClassifier(n_estimators=tree, bootstrap=False)
                rfc = rfc_model.fit(train_X, train_y)
                model.append(rfc)
                training_score[valid, 0] = i
                training_score[valid, 2:] = rfc.predict_proba(valid_X)

            # if len(categories) == 2:
            #     metrics= Metrics.getBinaryTaskMetrics(training_score[:, 3], training_score[:, 1])
            #     if metrics[6] > best_auc:
            #         best_auc = metrics[6]
            #         best_n_trees = tree
            #         best_model = model
            #         best_training_score = training_score
            # if len(categories) > 2:
            #     metrics = Metrics.getMutiTaskMetrics(training_score[:, 2:], training_score[:, 1])
            #     if metrics[0] > best_accuracy:
            #         best_accuracy = metrics[0]
            #         best_n_trees = tree
            #         best_model = model
            #         best_training_score = training_score

        self.training_score = pandas.DataFrame(best_training_score,
                                               columns=['Fold', 'Label'] + ['Score_%s' % i for i in categories])
        self.best_model = best_model
        self.best_n_trees = best_n_trees

class LightGBM(MachineLearning):
    def __init__(self, file):
        super(LightGBM, self).__init__(file)

    def LightGBM(self):
        fold = 5
        type = 'gbdt'
        parameters = {
            'num_leaves': list(range(20, 100, 10)),
            'max_depth': list(range(15, 55, 10)),
            'learning_rate': list(numpy.arange(0.01, 0.15, 0.02))
        }
        gbm = lightgbm.LGBMClassifier(boosting_type=type)
        gsearch = GridSearchCV(gbm, param_grid=parameters).fit(self.training_dataframe.values, self.training_datalabel)
        best_parameters = gsearch.best_params_
        num_leaves = best_parameters['num_leaves']
        max_depth = best_parameters['max_depth']
        learning_rate = best_parameters['learning_rate']

        categories = sorted(set(self.training_datalabel))
        X, y = self.training_dataframe.values, self.training_datalabel
        training_score = numpy.zeros((X.shape[0], len(categories) + 2))
        training_score[:, 1] = y
        model = []
        folds = StratifiedKFold(fold).split(X, y)
        for i, (train, valid) in enumerate(folds):
            train_X, train_y = X[train], y[train]
            valid_X, valid_y = X[valid], y[valid]
            gbm_model = lightgbm.LGBMClassifier(boosting_type=type, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate).fit(train_X, train_y)
            model.append(gbm_model)
            training_score[valid, 0] = i
            training_score[valid, 2:] = gbm_model.predict_proba(valid_X)
        self.training_score = pandas.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
        self.best_model = model


class LR(MachineLearning):
    def __init__(self, file):
        super(LR, self).__init__(file)

    def LR(self):
        categories = sorted(set(self.training_datalabel))
        X, y = self.training_dataframe.values, self.training_datalabel
        training_score = numpy.zeros((X.shape[0], len(categories) + 2))
        training_score[:, 1] = y
        model = []
        folds = StratifiedKFold(fold).split(X, y)
        for i, (train, valid) in enumerate(folds):
            train_X, train_y = X[train], y[train]
            valid_X, valid_y = X[valid], y[valid]
            lr_model = LogisticRegression(C=1.0, random_state=0).fit(train_X, train_y)
            model.append(lr_model)
            training_score[valid, 0] = i
            training_score[valid, 2:] = lr_model.predict_proba(valid_X)
        self.training_score = pandas.DataFrame(training_score,
                                               columns=['Fold', 'Label'] + ['Score_%s' % i for i in categories])
        self.best_model = model


class SVM(MachineLearning):
    def __init__(self, file):
        super(SVM, self).__init__(file)

    def SVM(self):
        kernel = 'poly'
        fold = 5
        gamma = 1 / self.training_dataframe.values.shape[1]
        penalityRange = (0.1, 15, 0.1)
        parameters = {'kernel': [kernel], 'C': penalityRange, 'gamma': 2.0 ** numpy.arange(0.001, 10)}

        optimizer = GridSearchCV(svm.SVC(probability=True), parameters).fit(self.training_dataframe.values,
                                                                            self.training_datalabel)
        params = optimizer.best_params_
        penality = params['C']
        gamma = params['gamma']

        categories = sorted(set(self.training_datalabel))
        X, y = self.training_dataframe.values, self.training_datalabel
        training_score = numpy.zeros((X.shape[0], len(categories) + 2))
        training_score[:, 1] = y
        model = []
        folds = StratifiedKFold(fold).split(X, y)
        for i, (train, valid) in enumerate(folds):
            train_X, train_y = X[train], y[train]
            valid_X, valid_y = X[valid], y[valid]
            svm_model = svm.SVC(C=penality, kernel=kernel, degree=3, gamma=gamma, coef0=0.0, shrinking=True,
                                probability=True, random_state=1)
            svc = svm_model.fit(train_X, train_y)
            model.append(svc)
            training_score[valid, 0] = i
            training_score[valid, 2:] = svc.predict_proba(valid_X)
        self.training_score = pandas.DataFrame(training_score,
                                               columns=['Fold', 'Label'] + ['Score_%s' % i for i in categories])
        self.best_model = model


class KNN(MachineLearning):
    def __init__(self, file):
        super(KNN, self).__init__(file)

    def KNN(self):
        fold = 5
        n_neighbors = 3

        categories = sorted(set(self.training_datalabel))
        X, y = self.training_dataframe.values, self.training_datalabel
        training_score = numpy.zeros((X.shape[0], len(categories) + 2))
        training_score[:, 1] = y
        model = []
        folds = StratifiedKFold(fold).split(X, y)
        for i, (train, valid) in enumerate(folds):
            train_X, train_y = X[train], y[train]
            valid_X, valid_y = X[valid], y[valid]
            knn_model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(train_X, train_y)
            model.append(knn_model)
            training_score[valid, 0] = i
            training_score[valid, 2:] = knn_model.predict_proba(valid_X)
        self.training_score = pandas.DataFrame(training_score,
                                               columns=['Fold', 'Label'] + ['Score_%s' % i for i in categories])
        self.best_model = model

f_path = os.path.join('.', 'training.csv')
f = MachineLearning(f_path)
f.data_import(f_path)
#f.RF()
#f.SVM()
#f.LR()
#f.LightGBM()
#f.KNN()