import os
import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot
from scikitplot.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor, XGBClassifier, plot_importance
import xgboost as xgb
from pycaret import classification
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer

warnings.filterwarnings('ignore')
xgb.set_config(verbosity=0)  # 忽略warnings
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']




def xgboost_parameters():
    """
    模型参数设置

    """

    # 1. n_estimators
    # 参数的最佳取值：{'n_estimators': 5}
    # 最佳模型得分: 0.5396603396603397
    # params = {'n_estimators': [5, 10, 50, 75, 100, 200]}

    # 2. min_child_weight[default=1],range: [0,∞] 和 max_depth[default=6],range: [0,∞]
    # 参数的最佳取值：{'max_depth': 2, 'min_child_weight': 3}
    # 最佳模型得分: 0.6072115384615385
    # params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}

    # 3. gamma[default=0, alias: min_split_loss],range: [0,∞]
    # 参数的最佳取值：{'gamma': }
    # 最佳模型得分: 0.6072115384615385
    # params = {'gamma': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

    # 4. subsample[default=1],range: (0,1] 和 colsample_bytree[default=1],range: (0,1]
    # 参数的最佳取值：{'colsample_bytree': , 'subsample': 0.4}
    # 最佳模型得分: 0.6319978632478633
    # params = {'subsample': [0.6, 0.7, 0.8, 0.9, 1], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]}

    # 5. alpha[default=0, alias: reg_alpha], 和 lambda[default=1, alias: reg_lambda]
    # 参数的最佳取值：{'reg_alpha': , 'reg_lambda': }
    # 最佳模型得分: 0.6319978632478633
    # params = {'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1, 2, 3], 'lambda': [0.05, 0.1, 1, 2, 3, 4]}

    # 6. learning_rate[default=0.3, alias: eta],range: [0,1]
    # 参数的最佳取值: {'learning_rate': 0.5}
    # 最佳模型得分:
    params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.25, 0.3, 0.4]}

    fine_params = {
                   'n_estimators': 5,
                   'max_depth': 2,
                   'min_child_weight': 3,
                   'subsample': 0.4,
                   'reg_lambda': 0.5
                   }

    return params, fine_params

def model_adjust_parameters(cv_params, other_params, X_train, y_train):
    """
        模型调参
    """
    # 模型基本参数
    model = XGBClassifier(**other_params)
    # 训练集5折交叉验证
    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1)
    # 模型训练
    optimized_param.fit(X_train, y_train)

    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean_score: %f,  params: %r" % (mean, param))

    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))

    print('最佳模型得分:{0}'.format(optimized_param.best_score_))

    parameters_score = pd.DataFrame(params, means)
    parameters_score['means_score'] = parameters_score.index
    parameters_score = parameters_score.reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(parameters_score.iloc[:, :-1], 'o-')
    plt.legend(parameters_score.columns.to_list()[:-1], loc='upper left')
    plt.title('Parameters_size', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.subplot(2, 1, 2)
    plt.plot(parameters_score.iloc[:, -1], 'r+-')
    plt.legend(parameters_score.columns.to_list()[-1:], loc='upper left')
    plt.title('Score', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.show()


def data_preprocessing(abd_path, meta_path, host_table, marker_path, host_name):
    """
    训练数据预处理
    """
    host_name_list = []
    with open(host_name, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            host_name_list.append(line.strip('\n'))

    df_tmp = pd.read_table(host_table)
    df_host = pd.DataFrame(columns=host_name_list)
    for i in host_name_list:
        df_host[i] = df_tmp[i]

    host_list = df_host.values.tolist()
    host_nparray = np.array(host_list)
    # print(host_list)

    # one-hot处理宿主变量
    integer_encodeds = []
    for i in host_nparray.T:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(i)
        integer_encodeds.append(integer_encoded)
    integer_encodeds = np.array(integer_encodeds).T
    enc = OneHotEncoder()
    enc.fit(integer_encodeds)
    host = enc.transform(integer_encodeds).toarray()
    # print(host)


    df = pd.read_table(abd_path)
    marker_list = []
    all_marker_list = []
    all_marker_list = df.columns.values.tolist()  # 全部菌列表
    del (all_marker_list[0])
    # print(all_marker_list)
    with open(marker_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            marker_list.append(line.strip('\n'))  # marker_list:特征列表

    marker_df = pd.DataFrame(columns=marker_list)
    for i in marker_list:
        marker_df[i] = df[i]

    X_value2 = marker_df.values.tolist()  # 选出特征的X
    # X_all = df.values.tolist()  # 全部特征X
    arr_2 = np.array(X_value2)
    scaler = preprocessing.MinMaxScaler()  # 归一化
    abd = scaler.fit_transform(arr_2)

    com = np.append(abd, host, axis=1)
    norm_com = preprocessing.normalize(com, norm='l2')  # 正则化
    X = norm_com.tolist()


    y = []
    with open(meta_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            value = line.split('\t')[1].strip()
            if value == 'High':
                y.append(1)
            else:
                y.append(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    onehot_hostname_list = ["Sex_0", "Sex_1", "Age_0", "Age_1", "Age_2", "BMI_0", "BMI_1", "BMI_2", "Probiotics_0",
                            "Probiotics_1", "Diseases_0", "Diseases_1"]
    feature_list = marker_list + onehot_hostname_list

    print(feature_list)

    return X, y, X_train, X_test, y_train, y_test, feature_list


def model_fit(X_train, X_test, y_train, y_test, feature):
    """
    模型训练
    """
    model = XGBClassifier(n_estimators=5,
                          max_depth=2,
                          min_child_weight=3,
                          subsample=0.4,
                          learning_rate=0.5
                          )



    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).tolist()
    print('y_test：', y_test)
    print('y_pred：', y_pred)

    y_pred_proba = model.predict_proba(X_test)
    y_pred_proba_ = []
    for i in y_pred_proba.tolist():
        y_pred_proba_.append(i[1])
    print('y_pred_proba_：', y_pred_proba_)  # 结果类别是1的概率

    metrics_sklearn(y_test, y_pred, y_pred_proba, y_pred_proba_)

    feature_importance_selected(model,feature)


    return model

def metrics_sklearn(y_test, y_pred, y_pred_proba, y_pred_proba_):

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy：%.4f' % (accuracy))

    precision = precision_score(y_test, y_pred)
    print('Precision：%.4f' % (precision))

    recall = recall_score(y_test, y_pred)
    print('Recall：%.4f' % (recall))

    f1 = f1_score(y_test, y_pred)
    print('F1：%.4f' % (f1))

    auc = roc_auc_score(y_test, y_pred_proba_)
    print('AUC：%.4f' % (auc))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="cornflowerblue",
        lw=lw,
        label="ROC curve (area = %.4f)" % auc,
    )
    plt.plot([0, 1], [0, 1], color="gainsboro", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("")
    plt.legend(loc="lower right")
    plt.savefig('D:\\work\\Se\\fig\\fig4-d.pdf')
    plt.show()


def feature_importance_selected(clf_model, feature_names):
    """
    模型特征重要性
    """
    clf_model.get_booster().feature_names = feature_names

    # print(feature_names)
    # plot_importance(clf_model).set_yticklabels([feature_names])
    plot_importance(clf_model
                    ,max_num_features=10
                    # ,importance_type='gain'
                    )
    plt.show()



def model_load(model_path, abd_path, meta_path, marker_path, host_name, host_table):
    """
    加载模型
    """
    host_name_list = []
    with open(host_name, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            host_name_list.append(line.strip('\n'))

    df_tmp = pd.read_table(host_table)
    df_host = pd.DataFrame(columns=host_name_list)
    for i in host_name_list:
        df_host[i] = df_tmp[i]

    host_list = df_host.values.tolist()
    host_nparray = np.array(host_list)

    # one-hot处理宿主变量
    integer_encodeds = []
    for i in host_nparray.T:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(i)
        integer_encodeds.append(integer_encoded)
    integer_encodeds = np.array(integer_encodeds).T
    enc = OneHotEncoder()
    enc.fit(integer_encodeds)
    host = enc.transform(integer_encodeds).toarray()

    df = pd.read_table(abd_path)
    marker_list = []
    all_marker_list = []
    all_marker_list = df.columns.values.tolist()  # 全部菌列表
    del (all_marker_list[0])
    # print(all_marker_list)
    with open(marker_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            marker_list.append(line.strip('\n'))  # marker_list:特征列表

    marker_df = pd.DataFrame(columns=marker_list)
    for i in marker_list:
        marker_df[i] = df[i]

    X_value2 = marker_df.values.tolist()  # 选出特征的X
    # X_all = df.values.tolist()  # 全部特征X
    arr_2 = np.array(X_value2)
    scaler = preprocessing.MinMaxScaler()  # 归一化
    abd = scaler.fit_transform(arr_2)

    com = np.append(abd, host, axis=1)  # 丰度与宿主变量合并
    norm_com = preprocessing.normalize(com, norm='l2')  # 正则化
    X = norm_com.tolist()
    print(X)


    y = []
    with open(meta_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            value = line.split('\t')[1].strip()
            if value == 'High':
                y.append(1)
            else:
                y.append(0)

    X_transform = np.array(X)
    model = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model(model_path)
    model._Booster = booster


    y_pred = model.predict(X).tolist()
    # y_test_ = y_test.values
    print('y_test：', y)
    print('y_pred：', y_pred)

    y_pred_proba = model.predict_proba(X)
    y_pred_proba_ = []
    for i in y_pred_proba.tolist():
        y_pred_proba_.append(i[1])
    print('y_pred_proba_：', y_pred_proba_)

    metrics_sklearn(y, y_pred, y_pred_proba, y_pred_proba_)



if __name__ == '__main__':

    marker_path = "data\\All\\Shi1_biomarker_genus(石家庄第一批特征).txt"  # 石家庄第一批特征
    marker_all_path = "data\\All\\new_all_biomarker_genus2.txt"  # 三批公共marker（3个）
    host_name_path = "data\\All\\host_feature_name.txt"  # 宿主变量名
    OTU_model_path = "model\\OTU_XGBoost_Classifier.model"  # OTU模型路径
    genus_model_path = "model\\genus_XGBoost_Classifier.model"  # 用石家庄第一批训练的genus模型路径

    Shi1_train_path = "data\\Shi_1\\156_before_taxa.genus.Abd"  # 训练样本
    Shi1_train_meta_path = "data\\Shi_1\\meta_156.txt"  # 标签
    Shi1_host_path = "data\\Shi_1\\Shi1_host.txt"  # 石家庄第一批宿主变量表
    Shi1_marker_new = "data\\Shi_1\\Shi1_biomarker_genus_new.txt"

    Shi2_test_path = "data\\Shi_2\\Shi2_taxa.genus.Abd"  # 石家庄队列测试样本
    Shi2_meta_path = "data\\Shi_2\\Shi2_meta.txt"  # 石家庄队列标签
    Shi2_host_path = "data\\Shi_2\\Shi2_host.txt"  # 石家庄第二批宿主变量表

    Su_test_path = "data\\Su\\Su_taxa.genus.Abd"  # 苏州队列测试样本
    Su_meta_path = "data\\Su\\Su_meta.txt"  # 苏州队列标签
    Su_host_path = "data\\Su\\Su_host.txt"  # 苏州队列宿主变量表


    all_abd = "data\\All\\three_batch_all_taxa.genus.Abd"  # 三队列全部样本丰度
    lable_region = "data\\All\\lable_region.txt"  # 地域标签
    marker_region = "data\\All\\marker_region.txt"  # 地域marker


    # 数据预处理，将数据划分为X_train, X_test, y_train, y_test
    X, y, X_train, X_test, y_train, y_test, feature_list = \
        data_preprocessing(Shi1_train_path, Shi1_train_meta_path, Shi1_host_path, marker_path, host_name_path)

    # # xgboost参数组合
    # adj_params, fixed_params = xgboost_parameters()
    # # 模型调参
    # model_adjust_parameters(adj_params, fixed_params, X_train, y_train)

    model=model_fit(X_train, X_test, y_train, y_test, feature_list)

    # # 模型保存
    # model.save_model(genus_model_path)
    # 模型加载预测石家庄第二批
    # model_load(genus_model_path, Shi2_test_path, Shi2_meta_path, marker_path, host_name_path, Shi2_host_path)
    # 模型加载预测苏州队列
    # model_load(genus_model_path, Su_test_path, Su_meta_path, marker_path, host_name_path, Su_host_path)








