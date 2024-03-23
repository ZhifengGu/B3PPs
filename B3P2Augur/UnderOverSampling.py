from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from collections import Counter

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import SMOTEN
# from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import BorderlineSMOTE
# from imblearn.over_sampling import SVMSMOTE
# from imblearn.over_sampling import KMeansSMOTE
# from imblearn.over_sampling import ADASYN

from imblearn.under_sampling import RandomUnderSampler
# from imblearn.under_sampling import ClusterCentroids
# from imblearn.under_sampling import NearMiss
# from imblearn.under_sampling import EditedNearestNeighbours
# from imblearn.under_sampling import RepeatedEditedNearestNeighbours
# from imblearn.under_sampling import AllKNN
# from imblearn.under_sampling import CondensedNearestNeighbour
# from imblearn.under_sampling import OneSidedSelection
# from imblearn.under_sampling import NeighbourhoodCleaningRule
# from imblearn.under_sampling import InstanceHardnessThreshold

# from imblearn.combine import SMOTEENN
# from imblearn.combine import SMOTETomek

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# from sklearn.ensemble import BaggingClassifier
# from imblearn.ensemble import BalancedBaggingClassifier
# from imblearn.ensemble import RUSBoostClassifier
# from imblearn.ensemble import EasyEnsembleClassifier
# from imblearn.ensemble import BalancedRandomForestClassifier

import pandas as pd

f = pd.read_csv('Training.csv')
X_resampled = f.iloc[:, 1:].values
y_resampled = f.iloc[:, 0].values

number = 215
rus = RandomUnderSampler(sampling_strategy={0: number}, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

# number = 376
# cc = ClusterCentroids(sampling_strategy={0: number},random_state=0)
# X_resampled, y_resampled = cc.fit_resample(X_resampled, y_resampled)


# oss = OneSidedSelection(random_state=42)
# X_resampled, y_resampled = oss.fit_resample(X_resampled, y_resampled)




# number = 273
# cnn = CondensedNearestNeighbour(random_state=42)
# X_resampled, y_resampled = cnn.fit_resample(X_resampled, y_resampled)
# number = 273
# rus = NearMiss(sampling_strategy={0: number}, version=3)
# X_resampled, y_resampled = rus.fit_resample(X, y)

# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

# adasyn = ADASYN()
# X_resampled, y_resampled = adasyn.fit_resample(X_resampled, y_resampled)

borderline_smote = BorderlineSMOTE()
X_resampled, y_resampled = borderline_smote.fit_resample(X_resampled, y_resampled)

pd.concat([pd.DataFrame(y_resampled), pd.DataFrame(X_resampled)], axis=1).to_csv('resampled.csv', header=False, index=False)