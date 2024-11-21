import numpy as np
import pandas as pd

import xgboost as xgb
from boruta import BorutaPy
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import pairwise_distances,roc_curve, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

### editted machine learning functions
## anova_ttest
def anova_ttest(X, y):
    if len(np.unique(y)) == 2:
        groups = []
        for i in np.unique(y): groups.append(X[y == i])
        f_statistic, anova_p = stats.ttest_ind(*groups)
        anova_index = abs(np.log10(anova_p))
    else:
        groups = []
        for i in np.unique(y):
            groups.append(X[y == i])
        f_statistic, anova_p = stats.f_oneway(*groups)
        anova_index = abs(np.log10(anova_p))

    return anova_index

## LDA
def LDA(X, y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    most_discriminative_features = lda.coef_
    lda_index = np.sum(np.abs(most_discriminative_features), axis=0)
    return lda_index

## Random Forest
def RF(X,y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    RF_index = rf.feature_importances_

    return RF_index

## SVM
def SVM(X, y):
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    svm_index = np.sum(np.abs(svm.coef_), axis=0)

    return svm_index

## logit
def logit(X, y):
    logreg = LogisticRegression()
    selector = SelectFromModel(estimator=logreg)
    selected_features = selector.fit_transform(X, y)
    logit_index = selector.estimator_.coef_[0]

    return logit_index

## decision tree
def decisiontree(X, y):
    tree = DecisionTreeClassifier()
    selector = SelectFromModel(estimator=tree)
    selected_features = selector.fit_transform(X, y)
    tree_index = selector.estimator_.feature_importances_

    return tree_index

## GMM
def GMM(X, y):
    gmm = GaussianMixture(n_components=4)
    gmm.fit(X)
    gmm_index = gmm.weights_

    return gmm_index

## KNN
def KNN(X, y):
    def relief(X, y, k):
        n_features = X.shape[1]
        feature_scores = np.zeros(n_features)

        for i in range(X.shape[0]-1):
            target_instance = X[i]
            same_class_indices = np.where(y == y[i])[0]
            different_class_indices = np.where(y != y[i])[0]

            # 计算与当前样本的距离
            distances = pairwise_distances(X[i+1].reshape(1, -1), X).flatten()

            # 计算与同类样本的平均距离
            same_class_distances = distances[same_class_indices]
            nearest_same_class_indices = np.argsort(same_class_distances)[1:k + 1]  # 排除自身
            nearest_same_class_avg_distance = np.mean(same_class_distances[nearest_same_class_indices])

            # 计算与异类样本的平均距离
            different_class_distances = distances[different_class_indices]
            nearest_different_class_indices = np.argsort(different_class_distances)[:k]
            nearest_different_class_avg_distance = np.mean(different_class_distances[nearest_different_class_indices])

            # 更新特征得分
            feature_scores += np.sum(np.abs(target_instance - X[same_class_indices[nearest_same_class_indices]]) / (
                        k * nearest_same_class_avg_distance), axis=0)
            feature_scores -= np.sum(
                np.abs(target_instance - X[different_class_indices[nearest_different_class_indices]]) / (
                            k * nearest_different_class_avg_distance), axis=0)

        return feature_scores

    KNN_index = np.abs(relief(X, y, k=5))

    return KNN_index

## PLSDA
def plsda(X, y):
    def compute_VIP(X, y, R, T, A):
        """
        计算模型中各预测变量的VIP值
        :param X: 数据集X
        :param y: 标签y
        :param R: A个PLS成分中，每个成分a都对应一套系数wa将X转换为成分得分，系数矩阵写作R，大小为p×A
        :param T: 得分矩阵记做T，大小为n×A，ta代表n个样本的第a个成分的得分列表
        :param A: PLS成分的总数
        :return: VIPs = np.zeros(p)
        """
        p = X.shape[1]
        Q2 = np.square(np.dot(y.T, T))

        VIPs = np.zeros(p)
        temp = np.zeros(A)
        for j in range(p):
            for a in range(A):
                temp[a] = Q2[a] * pow(R[j, a] / np.linalg.norm(R[:, a]), 2)
            VIPs[j] = np.sqrt(p * np.sum(temp) / np.sum(Q2))
        return VIPs

    n_component = 3
    model = PLSRegression(n_components=n_component)
    model.fit(X, y)
    x_test_trans = model.transform(X)
    plsda_index = compute_VIP(X, y, model.x_rotations_, x_test_trans, n_component)

    return plsda_index

## xgboost
def GMM(X, y):
    model = xgb.XGBClassifier()
    model.fit(X, y)
    xgboost_index = model.feature_importances_

    return xgboost_index

## boruta
def boruta(X, y):
    y_ = y.ravel()
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(X, y_)
    boruta_index = feat_selector.ranking_

    return boruta_index

## RFE
def RFE_ml(X, y):
    estimator = LogisticRegression()
    rfe = RFE(estimator, n_features_to_select=1)
    rfe.fit(X, y)
    rfe_index = [len(rfe.ranking_) + 1 - rank for rank in rfe.ranking_]

    return rfe_index

### Combined ML index
def Combined_ML_index(X, y,args):

    assert sum([args.ANOVA, args.LDA, args.RF, args.SVM, args.Logit, args.Tree, args.GMM, args.KNN, args.PLSDA, args.Xgboost, args.Boruta, args.RFE]) >= 4, \
        "The number of Machine learning indexes should >= 4."

    ### calculated machine learning index
    num_omics = X.shape[1]
    anova_index = anova_ttest(X, y) if args.ANOVA else np.zeros(num_omics)
    lda_index = LDA(X, y) if args.LDA else np.zeros(num_omics)
    RF_index = RF(X,y) if args.RF else np.zeros(num_omics)
    svm_index = SVM(X, y) if args.SVM else np.zeros(num_omics)
    logit_index = logit(X, y) if args.Logit else np.zeros(num_omics)
    tree_index = decisiontree(X, y) if args.Tree else np.zeros(num_omics)
    gmm_index = GMM(X, y) if args.GMM else np.zeros(num_omics)
    KNN_index = KNN(X, y) if args.KNN else np.zeros(num_omics)
    plsda_index = plsda(X, y) if args.PLSDA else np.zeros(num_omics)
    xgboost_index = GMM(X, y) if args.Xgboost else np.zeros(num_omics)
    boruta_index = boruta(X, y) if args.Boruta else np.zeros(num_omics)
    rfe_index = RFE_ml(X, y) if args.RFE else np.zeros(num_omics)

    # Combined results
    ML_index = pd.DataFrame({
        'ANOVA':list(anova_index),
        'LDA':list(np.abs(lda_index)),
        'RF':list(np.abs(RF_index)),
        'SVM':list(np.abs(svm_index)),
        'Logit':list(np.abs(logit_index)),
        'Tree':list(np.abs(tree_index)),
        'GMM':list(np.abs(gmm_index)),
        'KNN':list(np.abs(KNN_index)),
        'PLSDA':list(np.abs(plsda_index)),
        'Xgboost':list(np.abs(xgboost_index)),
        'Boruta':list(np.abs(boruta_index)),
        'RFE':list(np.abs(rfe_index))
    })

    return ML_index

## example
if __name__ == '__main__':
    ### load excample data
    iris = load_iris()
    X = iris.data
    y = iris.target

    ### Obtained and combined ML index
    ML_index = Combined_ML_index(X, y)
