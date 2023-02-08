import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor, XGBClassifier, plot_importance
import xgboost as xgb
from pycaret import classification
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer

from Se_Predict.final.Se_Predict_final import data_preprocessing, model_fit, metrics_sklearn

warnings.filterwarnings('ignore')
xgb.set_config(verbosity=0)  # 忽略warnings
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']



def shi2_model_fit(X_train, X_test, y_train, y_test, feature):

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
    print('y_pred_proba_：', y_pred_proba_)

    metrics_sklearn(y_test, y_pred, y_pred_proba, y_pred_proba_)

    return model



def shi12_train_shi2_30_test(Shi1_abd, Shi1_meta, Shi1_host, Shi2_abd, Shi2_meta, Shi2_host, marker_all, host_name):

    # Shi1_X_all: 石家庄第一批全部样本训练数据固定部分
    # Shi1_y_all: 石家庄第一批全部样本标签
    Shi1_X_all, Shi1_y_all, Shi1_X_train, Shi1_X_test, Shi1_y_train, Shi1_y_test, feature_list = \
        data_preprocessing(Shi1_abd, Shi1_meta, Shi1_host, marker_all, host_name)

    host_name_list = []
    with open(host_name, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            host_name_list.append(line.strip('\n'))

    df_tmp = pd.read_table(Shi2_host)
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

    df = pd.read_table(Shi2_abd)
    marker_list = []
    all_marker_list = []
    all_marker_list = df.columns.values.tolist()  # 全部菌列表
    del (all_marker_list[0])
    # print(all_marker_list)
    with open(marker_all, encoding='utf-8') as f:
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

    y = []
    with open(Shi2_meta, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            value = line.split('\t')[1].strip()
            if value == 'High':
                y.append(1)
            else:
                y.append(0)

    Shi2_X_train, Shi2_X_test, Shi2_y_train, Shi2_y_test = train_test_split(X
                                                                            ,y
                                                                            ,test_size=0.3
                                                                            ,random_state=0
                                                                            ,stratify=y
                                                                            )

    Shi2_X_train_True, Shi2_X_test_False, Shi2_y_train_True, Shi2_y_test_False = train_test_split(Shi2_X_train,
                                                                                                  Shi2_y_train,
                                                                                                  train_size=0.49,
                                                                                                  random_state=8
                                                                                                  )

    X_train_all = Shi1_X_all + Shi2_X_train_True
    y_train_all = Shi1_y_all + Shi2_y_train_True

    feature_list = marker_list + host_name_list

    model=shi2_model_fit(X_train_all, Shi2_X_test, y_train_all, Shi2_y_test, feature_list)




if __name__ == '__main__':

    marker_path = "data\\All\\Shi1_biomarker_genus(石家庄第一批特征).txt"  # 石家庄第一批特征
    marker_all_path = "data\\All\\new_all_biomarker_genus2.txt"  # 三批公共marker（3个）
    host_name_path = "data\\All\\host_feature_name.txt"  # 宿主变量名
    all_genus_model_path = "model\\all_genus_XGBoost_Classifier.model"  # genus模型路径

    Shi1_test_path = "data\\Shi_1\\156_before_taxa.genus.Abd"
    Shi1_meta_path = "data\\Shi_1\\meta_156.txt"
    Shi1_host_path = "data\\Shi_1\\Shi1_host.txt"  # 石家庄第一批宿主变量表

    Shi2_test_path = "data\\Shi_2\\Shi2_taxa.genus.Abd"  # 石家庄队列测试样本
    Shi2_meta_path = "data\\Shi_2\\Shi2_meta.txt"  # 石家庄队列标签
    Shi2_host_path = "data\\Shi_2\\Shi2_host.txt"  # 石家庄第二批宿主变量表

    Su_test_path = "data\\Su\\Su_taxa.genus.Abd"  # 苏州队列测试样本
    Su_meta_path = "data\\Su\\Su_meta.txt"  # 苏州队列标签
    Su_host_path = "data\\Su\\Su_host.txt"  # 苏州队列宿主变量表


    shi12_train_shi2_30_test(Shi1_test_path, Shi1_meta_path, Shi1_host_path,
                           Shi2_test_path, Shi2_meta_path, Shi2_host_path,
                           marker_path, host_name_path)