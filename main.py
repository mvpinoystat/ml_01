#!/home/pinoystat/Documents/python/environment/datascience/bin/python
# Submission 7 Leaderboard score: 0.71

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import log_loss
import numpy as np
import scipy.stats as st

# import graphs
# from preprocessing import generateUniqueDatasets, generateUniqueIndices
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

import sys
# from graphs import *
# sns.set()

NUMERIC_PREDICTORS = np.array(['AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC',
                               'BD ', 'BN', 'BP', 'BQ', 'BR', 'BZ', 'CB', 'CC', 'CD ',
                               'CF', 'CH', 'CL', 'CR', 'CS', 'CU', 'CW ', 'DA', 'DE', 'DF',
                               'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY', 'EB', 'EE', 'EG', 'EH',
                               'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI', 'FL', 'FR', 'FS',
                               'GB', 'GE', 'GF', 'GH', 'GI', 'GL'])

SIG_FEATURES = np.array([
    "AB", "AF", "AM", "AR", "BC", "BN", "BP", "BQ", "BR",
    "CC", "CD ", "CF", "CR", "CW ", "DA", "DE", "DF",
    "DH", "DI", "DN", "DU", "DY", "EB", "EE", "EH",
    "FD ", "FE", "FI", "FL", "FR", "GF", "GL", "BZ"
])

B_PREDICTORS = np.array([
    "AF", "AM", "BC", "BN", "BQ", "CR", "CW ",
    "DI", "DU", "EH", "FD ", "FE", "FI", "FL", "FR", "GL"
])

G_PREDICTORS = np.array([
    "AB", "AF", "AM", "BC", "BP", "BR",
    "CC", "CD ", "CF", "CR", "CW ", "DA",
    "DH", "DI", "EB", "EE", "EH", "FE",
    "FI", "FR", "BZ"
])

D_PREDICTORS = np.array([
    "AB", "AF", "AM", "AR", "BC", "BP",
    "BQ", "BR", "CD ", "CF", "CR", "DA",
    "DH", "DI", "FI", "FR", "BZ"
])

"""
Below are the transformers used in Submission 3 and up
"""


class InitialImpute(BaseEstimator, TransformerMixin):
    """
    Initial imputation based on EDA (Initial use was during Submission 3)
    """
    impute_parameter_list = [
        [580, 'BR', 633.197222],
        [155, 'BR', 579.990218],
        [378, 'BZ', 50092.459300 / 10],
    ]
    xx: pd.DataFrame
    xx_class: pd.Series

    # EJ is not included even though it is significant
    # BZ is moved on the last column for easy cutting
    significant_features = SIG_FEATURES

    def fit(self, X):
        self.xx = X.copy()
        return self

    def transform(self, X):
        # do the imputation now:
        self.imputation()
        self.imputationBQ()
        return self.xx[self.significant_features].values

    def imputation(self):
        for index, feature, value in self.impute_parameter_list:
            self.xx.loc[index, feature] = value
        return self

    def imputationBQ(self):
        v = self.xx[self.xx.BQ.isnull()].index.to_list()
        self.xx.loc[v, 'BQ'] = 53.775015
        return self


class PowerTransformerCustom(BaseEstimator, TransformerMixin):
    z: PowerTransformer
    train_class: np.array
    xx: np.array

    def __init__(self, train_data: np.array, sig_features):
        self.z = PowerTransformer()
        self.train_class = train_data
        self.significant_features = sig_features

    def fit(self, X, y=0):
        # X contains 33 rows of numeric data. Columns defined
        # in InitialImpute. Fit only those with == 1.
        q = pd.DataFrame(data=X, columns=self.significant_features)
        m = q[self.significant_features].copy()
        # fit now:
        self.z.fit(m.values)
        return self

    def transform(self, X):
        return self.z.transform(X)


class CategorizeFeature(BaseEstimator, TransformerMixin):
    """
    Note: feature BZ with numeric index 9 needs to be divided into 2 intervals only. The rest are 4.
    BZ is in the last column
    """
    train: pd.DataFrame
    xx: pd.DataFrame
    intervals: np.array
    BZ_interval: np.array

    # probabilities
    probability_bins = []

    # output list:
    output = []
    final_output: np.array
    divisions = 0

    def __init__(self, train_data: pd.DataFrame, div=6):
        self.train = train_data
        self.divisions = div
        self.probability_bins = []

    def getIntervals(self):
        # Subtract 1 since the last column needs to be cut into 2 bins only
        for i in range(self.xx.shape[1] - 1):
            z = pd.cut(self.xx[:, i], bins=self.divisions, retbins=True)
            # Store the intervals cross-tabulation
            box = pd.DataFrame(data=z[0], columns=["intervals"])
            box['data'] = self.xx[:, i]
            box['Class'] = self.train.Class.values
            cross = pd.crosstab(index=box.intervals, columns=box.Class,
                                values=box.data, aggfunc="count")
            cross = cross / cross.sum()
            self.probability_bins.append(cross)

            if i == 0:
                self.intervals = z[1]
            else:
                self.intervals = np.c_[self.intervals, z[1]]
        # For the last bin:
        self.BZ_interval = pd.cut(self.xx[:, -1], bins=2, retbins=True)
        boxBZ = pd.DataFrame(data=self.xx[:, -1], columns=["data"])
        boxBZ['intervals'] = self.BZ_interval[0]
        boxBZ['Class'] = self.train.Class.values
        cross = pd.crosstab(index=boxBZ.intervals, columns=boxBZ.Class,
                            values=boxBZ.data, aggfunc="count")
        cross = cross / cross.sum()
        self.probability_bins.append(cross)
        return self

    def fit(self, X, y=0):
        self.xx = X
        # Get the cutting intervals for
        self.getIntervals()
        return self

    def transform(self, X):
        # reset the array!!:
        self.output = []

        # For each column, Identify which interval the values
        # should be binned and put in the accompanying 0 and 1 probabilities
        for columns in range(X.shape[1] - 1):
            container = np.zeros(shape=(X.shape[0], 2))  # renew container each column
            for rows in range(X.shape[0]):
                flag = 0
                # If division is 4, division 1, 2, and 3 will be used of 0,1,2,3,4. There are 5 cuts
                for i in range(self.divisions - 1):
                    if X[rows, columns] <= self.intervals[i + 1, columns]:
                        container[rows, 0] = self.probability_bins[columns].iloc[i, 0]
                        container[rows, 1] = self.probability_bins[columns].iloc[i, 1]
                        flag = 1
                        break  # Break the loop since the checking is already satisfied

                # This is for the last division , the greater than part.
                # Example: If division is 4, use the 3rd division and execute greater than
                if flag != 1:
                    container[rows, 0] = self.probability_bins[columns].iloc[self.divisions - 1, 0]
                    container[rows, 1] = self.probability_bins[columns].iloc[self.divisions - 1, 1]

            self.output.append(container)

        # For BZ:
        container = np.zeros(shape=(X.shape[0], 2))  # renew container each column
        for rows in range(X.shape[0]):
            if X[rows, -1] <= self.intervals[1, -1]:
                container[rows, 0] = self.probability_bins[len(self.probability_bins) - 1].iloc[0, 0]
                container[rows, 1] = self.probability_bins[len(self.probability_bins) - 1].iloc[0, 1]

            else:
                container[rows, 0] = self.probability_bins[len(self.probability_bins) - 1].iloc[1, 0]
                container[rows, 1] = self.probability_bins[len(self.probability_bins) - 1].iloc[1, 1]

        self.output.append(container)

        # arrange for final output:
        for items in range(len(self.output)):
            if items == 0:
                self.final_output = self.output[items]
            else:
                self.final_output = np.c_[self.final_output, self.output[items]]

        # testing:
        # print(self.final_output)
        return self.final_output

class RatioFeature(BaseEstimator, TransformerMixin):
    numerators = []
    denominators = []
    sig_features: np.array
    train_data: pd.DataFrame
    output: np.array
    age_conditions: np.array

    def __init__(self, train_data: pd.DataFrame, sig_features: np.array, age_conditions:np.array):
        self.train_data = train_data
        self.sig_features = sig_features
        self.numerators = []
        self.denominators = []
        self.age_conditions = age_conditions


    def fit(self, X, y=0):
        self.train_data['Class'] = self.age_conditions
        for i in self.sig_features:
            for j in self.sig_features:
                if i != j:
                    self.train_data['N'] = self.train_data[i]/self.train_data[j]
                    t = ttest_ind(self.train_data['N'][self.train_data.Class == 0],
                                  self.train_data['N'][self.train_data.Class == 1])
                    if t.pvalue < 0.001:
                        print("significant: {}/{},sig={}".format(i,j, t.pvalue))
                        self.numerators.append(i)
                        self.denominators.append(j)
        print(len(self.numerators))
        print(len(self.denominators))
        return self

    def transform(self, X):
        # print("X shape: {}".format(X.shape))
        df = pd.DataFrame(data=X, columns=self.sig_features)
        for i in range(len(self.numerators)):
            xx = self.numerators[i]
            yy = self.denominators[i]
            zz = df[xx]/df[yy]
            if i == 0:
                self.output = zz
            else:
                self.output = np.c_[self.output, zz.values]
        # print(self.output)
        return self.output

def JoinDataset():
    greeks = pd.read_csv("kaggle/input/icr-identify-age-related-conditions/greeks.csv")
    train = pd.read_csv("kaggle/input/icr-identify-age-related-conditions/train.csv")

    # greeks = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/greeks.csv")
    # train = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")
    combined = train.join(greeks.set_index("Id"), on="Id", how='left', lsuffix="training", rsuffix="greeks")
    return combined


# Analysis and check of feature importances
# For submission_7

def BuildModel(seed=7, feat_divisions=6, n_iterations=10):
    """
    Note: the best center is 2 or on the whole variables and not only using  1 or 0 when fitting the PowerTransform
    Note: the best division is 6 divisions.
    param feat_divisions0
    """
    mdat = JoinDataset()
    # fit the model:
    # Step 1. clean the data by putting in missing values:
    mdat2 = InitialImpute().fit_transform(mdat)
    mdat3 = pd.DataFrame(data=mdat2.copy(), columns=SIG_FEATURES)
    mdat3["Class"] = mdat.Class

    poly_param = {"kernel": ["poly"], "coef0": np.arange(0, 10, 1),
                  "degree": np.arange(0, 20),
                  "C": np.linspace(.001, 10000, 100000)}
    rbf_param = {"kernel": ["rbf"], "gamma": np.linspace(1E-2, 1.0, 1000),
                 "C": np.linspace(1, 1E4, 1000)}

    forest_config = {"max_features": [i for i in range(1, 100)],
                     "max_leaf_nodes": [i for i in range(2, NUMERIC_PREDICTORS.shape[0])],
                     "n_estimators": [i for i in range(3, 200)],
                     "criterion": ["gini", "entropy", "log_loss"]
                     }
    ada_param = {"n_estimators": [i for i in range(50, 101)],
                 "learning_rate": np.linspace(0.01, 20, 1000)}

    voting_param = {'rbf__C': np.linspace(1, 1E4, 100), 'rbf__kernel':['rbf'],
                    "rbf__gamma": np.linspace(1E-2, 1.0, 1000),
                    'poly__kernel': ['poly'], 'poly__coef0': np.arange(0,10,1),
                    'poly__degree': np.arange(0,20), 'poly__C': np.linspace(.001, 10000, 100000),
                    'forest__max_features': [i for i in range(1,100)],
                    'forest__max_leaf_nodes': [i for i in range(2, NUMERIC_PREDICTORS.shape[0])],
                    'forest__n_estimators': [i for i in range(3, 200)],
                    'forest__criterion': ['gini', 'entropy', 'log_loss']}
    # algo = RandomForestClassifier(random_state=seed)
    # algo = SVC(probability=True, random_state=seed)
    # algo = AdaBoostClassifier(estimator=SVC(C=3283.954954954955, gamma=0.6036036036036037, probability=True,
    # random_state=seed), random_state=seed)
    clf1 = SVC(probability=True,random_state=seed)

    clf2 = SVC(probability=True,random_state=seed)

    clf3 = RandomForestClassifier(random_state=seed)

    # clf4 = GaussianNB()
    # clf5 = QuadraticDiscriminantAnalysis()
    algo = VotingClassifier(
        estimators=[('rbf', clf1), ('poly', clf2), ('forest', clf3)],
        voting="soft"
    )

    # Here ony clf2 was used
    model = RandomizedSearchCV(estimator=clf2, param_distributions=poly_param,
                               n_iter=n_iterations, n_jobs=-1, random_state=seed, scoring="neg_log_loss")

    process = Pipeline([("si", SimpleImputer(strategy="median")),
                        ("pt", PowerTransformerCustom(train_data=mdat3,
                                                      sig_features=SIG_FEATURES)),
                        # ("ratio", RatioFeature(train_data=mdat3, sig_features=SIG_FEATURES,
                        #                       age_conditions=mdat.Class.values)),
                        ("cf", CategorizeFeature(train_data=mdat3, div=feat_divisions)),
                        ("model", model)])

    # Predict the test set:
    process.fit(mdat3[SIG_FEATURES].values, mdat3.Class.values)
    cv_results_ = process.named_steps['model'].cv_results_
    output = pd.DataFrame(cv_results_['params'])
    output['log_loss'] = -1 * cv_results_['mean_test_score']
    print("best estimator log loss:", end="")
    print(np.min(output.log_loss))

    # Best estimators:
    """
    best estimator:
    SVC(C=3283.954954954955, gamma=0.6036036036036037, probability=True,
    random_state=7)
    0.239304 -> log loss
    best estimator:
    RandomForestClassifier(criterion='entropy', max_features=64, max_leaf_nodes=40,
                       n_estimators=76, random_state=7)
    0.248711 -> log loss
    best estimator:
    SVC(C=3064.4313378633788, coef0=9, degree=8, kernel='poly', probability=True,
    random_state=7)
    0.243375 -> log loss
    
    VotingClassifier(estimators=[('rbf',
                              SVC(C=3637.0, gamma=0.9336036036036037,  
                                  probability=True, random_state=7)),  
                             ('poly',
                              SVC(C=7594.676187291872, coef0=9, degree=16,
                                  kernel='poly', probability=True,
                                  random_state=7)),
                             ('forest',
                              RandomForestClassifier(max_features=68,  
                                                     max_leaf_nodes=23,
                                                     n_estimators=138, 
                                                     random_state=7))],
                 voting='soft')
     0.227335 -> log loss
    """

    # test_file_link = "/kaggle/input/icr-identify-age-related-conditions/test.csv"
    test_file_link = "kaggle/input/icr-identify-age-related-conditions/test.csv"
    test_file = pd.read_csv(test_file_link)

    # # transform and predict:

    test_predictions = process.predict_proba(test_file[SIG_FEATURES].values)
    # # submission_name = "/kaggle/working/submission.csv"
    submission_name = "kaggle/working/submission.csv"
    test_submission = pd.DataFrame({
        "Id": test_file.Id,
        "class_0": test_predictions[:, 0],
        "class_1": test_predictions[:, 1]
    })
    test_submission.to_csv(submission_name, index=False)
    print("Submission completed.")
    print(test_submission)

def plotData(data_, filename_, title: str):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.hist(data_)
    ax.set_title(title)
    plt.savefig(filename_)



if __name__ == "__main__":
    # model with div = 600 , submission = 9, leaderboard = 0.85
    div = 6
    # div = int(sys.argv[1])
    # print("division: {}".format(div))
    BuildModel(seed=0, n_iterations=10, feat_divisions=div)

    # BuildModel(seed=0)
    # mdat = JoinDataset()
    # fit the model:
    # Step 1. clean the data by putting in missing values:
    # mdat2 = InitialImpute().fit_transform(mdat)
    # mdat3 = pd.DataFrame(data=mdat2.copy(), columns=SIG_FEATURES)
    # print(mdat3)
    # print(mdat3.shape)


    # process = Pipeline([("si", SimpleImputer(strategy="median")),
    #                     ("pt", PowerTransformerCustom(train_data=mdat3,
    #                                                   sig_features=SIG_FEATURES)),
    #                     ("ratio", RatioFeature(train_data=mdat3, sig_features=SIG_FEATURES,
    #                                            age_conditions=mdat.Class.values))])

    # q = process.fit_transform(mdat3)
    # print(q)
    # print(q.shape)


