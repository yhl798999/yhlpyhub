# %%
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as sm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import scale,StandardScaler
#%%
# 数据预处理
dataset = pd.read_csv("dataset121804.csv")
dataset.iloc[:, 0:5] = dataset.iloc[:, 0:5].astype('category')
scaler = StandardScaler().fit(dataset.iloc[:,5:14])
dataset.iloc[:,5:14] = scaler.transform(dataset.iloc[:,5:14])
dataset.dropna(inplace=True)
x = dataset.iloc[:, 1:14]
y = dataset.kernel
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=666)
#%%
# 预测数据读取
suijiset = pd.read_csv("suijidian.csv")
suijiset.iloc[:, 1:5] = suijiset.iloc[:, 1:5].astype('category')
suijiset.iloc[:, 5:14] = scaler.transform(suijiset.iloc[:,5:14])
x_suiji = suijiset.iloc[:, 1:14]
print(x_suiji.isnull().any())
# %%
# 逻辑回归模型，MvM模式，权重自平衡，算法是newton-cg，使用l2范数，其系数为1.5
logreg = LogisticRegression(multi_class="multinomial", solver="newton-cg", random_state=123,
                            class_weight='balanced', max_iter=1000, verbose=1, n_jobs=-1, C=1)
logreg.fit(x_train, y_train)
logreg_pred_proba_train = logreg.predict_proba(x_train)
logreg_pred_class_train = logreg.predict(x_train)
logreg_pred_proba_test = logreg.predict_proba(x_test)
logreg_pred_class_test = logreg.predict(x_test)

logreg_report_train = sm.classification_report(
    y_train, logreg_pred_class_train)
logreg_confusion_matrix_train = sm.confusion_matrix(
    y_train, logreg_pred_class_train)
logreg_report_test = sm.classification_report(
    y_test, logreg_pred_class_test)
logreg_confusion_matrix_test = sm.confusion_matrix(
    y_test, logreg_pred_class_test)

logreg_train_data_out = pd.DataFrame({'y_class': y_train, 'y_pred_class': logreg_pred_class_train,
                                      'y_prob_1': logreg_pred_proba_train[:, 0], 'y_prob_2': logreg_pred_proba_train[:, 1],
                                      'y_prob_3': logreg_pred_proba_train[:, 2], 'y_prob_4': logreg_pred_proba_train[:, 3]})
logreg_test_data_out = pd.DataFrame({'y_class': y_test, 'y_pred_class': logreg_pred_class_test,
                                      'y_prob_1': logreg_pred_proba_test[:, 0], 'y_prob_2': logreg_pred_proba_test[:, 1],
                                      'y_prob_3': logreg_pred_proba_test[:, 2], 'y_prob_4': logreg_pred_proba_test[:, 3]})
logreg_train_data_out.to_csv('logreg_train_data_out_0617.csv', sep=',', index=False, header=True)
logreg_test_data_out.to_csv('logreg_test_data_out_0617.csv', sep=',', index=False, header=True)

# %%
# 多层感知器模型
mlp = MLPClassifier(random_state=666,alpha=0.001,hidden_layer_sizes=(100,100))
mlp.fit(x_train,y_train)
mlp_pred_proba_train = mlp.predict_proba(x_train)
mlp_pred_class_train = mlp.predict(x_train)
mlp_pred_proba_test = mlp.predict_proba(x_test)
mlp_pred_class_test = mlp.predict(x_test)

mlp_report_train = sm.classification_report(y_true=y_train, y_pred=mlp_pred_class_train)
mlp_confusion_matrix_train = sm.confusion_matrix(y_true=y_train, y_pred=mlp_pred_class_train)
mlp_report_test = sm.classification_report(y_true=y_test, y_pred=mlp_pred_class_test)
mlp_confusion_matrix_test = sm.confusion_matrix(y_true=y_test, y_pred=mlp_pred_class_test)

mlp_train_data_out = pd.DataFrame({'y_class': y_train, 'y_pred_class': mlp_pred_class_train,
                                      'y_prob_1': mlp_pred_proba_train[:, 0], 'y_prob_2': mlp_pred_proba_train[:, 1],
                                      'y_prob_3': mlp_pred_proba_train[:, 2], 'y_prob_4': mlp_pred_proba_train[:, 3]})
mlp_test_data_out = pd.DataFrame({'y_class': y_test, 'y_pred_class': mlp_pred_class_test,
                                      'y_prob_1': mlp_pred_proba_test[:, 0], 'y_prob_2': mlp_pred_proba_test[:, 1],
                                      'y_prob_3': mlp_pred_proba_test[:, 2], 'y_prob_4': mlp_pred_proba_test[:, 3]})
mlp_train_data_out.to_csv('mlp_train_data_out_0617.csv', sep=',', index=False, header=True)
mlp_test_data_out.to_csv('mlp_test_data_out_0617.csv', sep=',', index=False, header=True)
# %%
# 随机森林模型
rfc = RandomForestClassifier(n_estimators=300, random_state=111, n_jobs=-1,
                             verbose=1, oob_score=True)
rfc.fit(x_train, y_train)
rfc_pred_proba_train = rfc.predict_proba(x_train)
rfc_pred_class_train = rfc.predict(x_train)
rfc_pred_proba_test = rfc.predict_proba(x_test)
rfc_pred_class_test = rfc.predict(x_test)

rfc_report_train = sm.classification_report(y_true=y_train, y_pred=rfc_pred_class_train)
rfc_confusion_matrix_train = sm.confusion_matrix(y_true=y_train, y_pred=rfc_pred_class_train)
rfc_report_test = sm.classification_report(y_true=y_test, y_pred=rfc_pred_class_test)
rfc_confusion_matrix_test = sm.confusion_matrix(y_true=y_test, y_pred=rfc_pred_class_test)

rfc_train_data_out = pd.DataFrame({'y_class': y_train, 'y_pred_class': rfc_pred_class_train,
                                      'y_prob_1': rfc_pred_proba_train[:, 0], 'y_prob_2': rfc_pred_proba_train[:, 1],
                                      'y_prob_3': rfc_pred_proba_train[:, 2], 'y_prob_4': rfc_pred_proba_train[:, 3]})
rfc_test_data_out = pd.DataFrame({'y_class': y_test, 'y_pred_class': rfc_pred_class_test,
                                      'y_prob_1': rfc_pred_proba_test[:, 0], 'y_prob_2': rfc_pred_proba_test[:, 1],
                                      'y_prob_3': rfc_pred_proba_test[:, 2], 'y_prob_4': rfc_pred_proba_test[:, 3]})
rfc_train_data_out.to_csv('rfc_train_data_out_0617.csv', sep=',', index=False, header=True)
rfc_test_data_out.to_csv('rfc_test_data_out_0617.csv', sep=',', index=False, header=True)
# %%
# 测试数据
logreg_pred_proba_suiji = logreg.predict_proba(x_suiji)
mlp_pred_proba_suiji = mlp.predict_proba(x_suiji)
rfc_pred_proba_suiji = rfc.predict_proba(x_suiji)
suiji_data_out = pd.DataFrame({'y_prob_logreg_1': logreg_pred_proba_suiji[:, 0], 'y_prob_logreg_2': logreg_pred_proba_suiji[:, 1],
                               'y_prob_logreg_3': logreg_pred_proba_suiji[:, 2], 'y_prob_logreg_4': logreg_pred_proba_suiji[:, 3],
                               'y_prob_mlp_1': mlp_pred_proba_suiji[:, 0], 'y_prob_mlp_2': mlp_pred_proba_suiji[:, 1],
                               'y_prob_mlp_3': mlp_pred_proba_suiji[:, 2], 'y_prob_mlp_4': mlp_pred_proba_suiji[:, 3],
                               'y_prob_rfc_1': rfc_pred_proba_suiji[:, 0], 'y_prob_rfc_2': rfc_pred_proba_suiji[:, 1],
                               'y_prob_rfc_3': rfc_pred_proba_suiji[:, 2], 'y_prob_rfc_4': rfc_pred_proba_suiji[:, 3]})
suiji_data_out.to_csv('preprob_suiji_0617.csv',sep=',', index=False, header=True)
#%%
print(logreg_report_train)
print(logreg_report_test)
print(logreg_confusion_matrix_train)
print(logreg_confusion_matrix_test)
#%%
print(mlp_report_train)
print(mlp_report_test)
print(mlp_confusion_matrix_train)
print(mlp_confusion_matrix_test)
print(rfc_report_train)
print(rfc_report_test)
print(rfc_confusion_matrix_train)
print(rfc_confusion_matrix_test)


# %%
