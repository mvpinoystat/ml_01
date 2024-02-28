import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import transformers as tx
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
from sklearn.pipeline import make_pipeline
import os
from scipy.stats import tukey_hsd
def centeringTest():
    """
    This tests if during power transformation and standardization, is fitting the standardization
    on data with class is 1 only or with 0 only or with both?
    Display: Tukey result showing that centering on all data (both 0 and 1) yielded the lowest
    log loss on average
    :return: none
    """
    os.system("clear")
    tt = pd.read_csv("output/submission_4/centering_profile.csv")
    print(tt.head(4))
    output = pd.pivot_table(data=tt, index="center", values="loss", aggfunc=np.mean)
    print(output)
    res = tukey_hsd(tt.loss[tt.center == 0].values, tt.loss[tt.center == 1].values,
                    tt.loss[tt.center == 2].values)
    print(res)


def JoinDataset():
    greeks = pd.read_csv("kaggle/input/icr-identify-age-related-conditions/greeks.csv")
    train = pd.read_csv("kaggle/input/icr-identify-age-related-conditions/train.csv")

    combined = train.join(greeks.set_index("Id"), on="Id", how='left', lsuffix="training", rsuffix="greeks")
    return combined


def checkClasses(joined: pd.DataFrame):
    # Check if the class are correct:
    checking = pd.crosstab(index=joined.Alpha, columns=joined.Class, values=joined.Alpha, aggfunc='count')
    print(checking)

    print("Checking variable EJ")
    var_ej = pd.crosstab(index=joined.Class, columns=joined.EJ, values=joined.EJ, aggfunc="count")
    print(var_ej)
    chi, p_value = stats.chisquare(var_ej)
    print("chi-square stat: {}, p_value = {}".format(chi, p_value))


def generateTrainData(random_state_=23):
    """
    Use to generate split train and test data using StratifiedShuffleSplit
    :return:
    """
    # Load the train data:
    train_data = pd.read_csv("kaggle/input/icr-identify-age-related-conditions/train.csv")

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.4, random_state=random_state_)
    v = list(splitter.split(train_data, train_data.Class))
    x_train = train_data.iloc[v[0][0], :]
    x_test = train_data.iloc[v[0][1], :]

    return [x_train, x_test]


def generateUniqueDatasets(random_state=0):
    """
    :return:A list of datasets
    """
    mdat = JoinDataset()
    if random_state == 0:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8)
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)

    v = splitter.split(mdat, y=mdat.Class)
    v = list(v)
    return [mdat.iloc[v[0][0], :], mdat.iloc[v[0][1], :]]

def generateUniqueIndices(random_state=0):
    mdat = JoinDataset()

    if random_state == 0:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8)
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)

    v = splitter.split(mdat, y=mdat.Class)
    return list(v)


def checkDistributionOfNumericPredictors():
    """
    Here, we check all numeric factors if they are normally distributed
    Note: All 55 numeric predictors are not normally distributed.
    """
    filename = "output/normality_test.csv"
    mdat = JoinDataset()
    factors = tx.NUMERIC_PREDICTORS
    file = open("output/normality_test.csv", "w")
    file.write("variable,min,median,max,shapiro_p_value\n")
    for c in factors:
        numbers = mdat[c]
        shapiro_stats = stats.shapiro(numbers)
        p_value = np.round(shapiro_stats.pvalue, 4)
        minimum = np.min(numbers)
        median = np.median(numbers)
        max = np.max(numbers)

        file.write(f"{c},{minimum},{median},{max},{p_value}\n")

    file.close()

    ddf = pd.read_csv(filename)
    print(ddf.sort_values(by="shapiro_p_value", ascending=True))


def getPossibleNormalFeatures():
    """
    Transform numeric predictors and check normality
    :return: Numeric features which can be reshaped into a normal distribution
    """
    datasets = generateUniqueDatasets()
    # dataset 0 is train:
    train = datasets[0]
    train = train[tx.NUMERIC_PREDICTORS].copy()  # Get only the numeric predictors to test
    pipe = make_pipeline(PowerTransformer(standardize=True))
    train = pipe.fit_transform(train)
    # Test now:
    p_value_list = []
    for c in range(train.shape[1]):
        result = stats.shapiro(train[:, c])
        p_value_list.append(np.round(result.pvalue, 4))

    p_value_list = np.array(p_value_list)
    results = pd.DataFrame({"feature": np.array(tx.NUMERIC_PREDICTORS),"p_value": p_value_list})
    significant = results[results['p_value'] > 0.05].copy()
    significant.to_csv("output/significant_normality_after_transform.csv", index=False)
    # print(significant)

    return significant


def showGreekTables():
    mdat = JoinDataset()
    print("Alpha and Beta")
    print("The number of Beta types in each Alpha type")
    alpha_beta = pd.crosstab(index=mdat.Alpha, columns=mdat.Beta, values=mdat.Beta, aggfunc="count")
    print(alpha_beta)
    print("Alpha and Gamma")
    print("The number of Gamma types in each Alpha type")
    alpha_gamma = pd.crosstab(index=mdat.Alpha, columns=mdat.Gamma, values=mdat.Gamma, aggfunc="count")
    print(alpha_gamma)
    print("Alpha and Delta")
    print("The number of Delta types in each Alpha type")
    alpha_delta = pd.crosstab(index=mdat.Alpha, columns=mdat.Delta, values=mdat.Delta, aggfunc="count")
    print(alpha_delta)


def saveFinalPredictors():
    datasets = generateTrainData()
    train = datasets[0].copy()
    # print(train.Class)
    test = datasets[1].copy()
    del datasets

    fu = make_pipeline(PowerTransformer(standardize=True),
                       FunctionTransformer(func=tx.squared))
    x_train = fu.fit_transform(train[tx.NUMERIC_PREDICTORS])
    x_train_df = pd.DataFrame(data=x_train, columns=tx.NUMERIC_PREDICTORS)
    filename = "output/feature_squared_diff.csv"
    dd = pd.read_csv(filename)
    d1 = dd[dd.abs_median_diff == 2.6599].copy()
    del dd
    selected_predictors = d1.varB_squared.values
    selected_predictors = np.append(selected_predictors, d1.varA_squared[0])
    print(selected_predictors)
    fName = "output/predictors_v2.csv"
    pd.DataFrame({"predictors": selected_predictors}).to_csv(fName, index=False)

def check_significant_at_alpha(alpha="B"):
    filename = "output/submission_2/significant_{}_in_alpha.csv".format(alpha)
    dd = pd.read_csv(filename)
    ordered = dd.sort_values(by="p_value", ignore_index=True)
    print(ordered[ordered.significant == True])
