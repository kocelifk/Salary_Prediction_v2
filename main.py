"""
###################################################
# PROJECT: SALARY PREDICTİON WITH MACHINE LEARNING
###################################################

# İş Problemi

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
# 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.


# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör
"""

###################################################
# GÖREV: Veri ön işleme ve özellik mühendisliği tekniklerini kullanarak maaş tahmin modeli geliştiriniz.
###################################################

############################################
# Gerekli Kütüphane ve Fonksiyonlar
############################################

import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from helpers.data_prep import *
from helpers.eda import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Tum Base Modeller
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df = pd.read_csv("datasets/hitters.csv")

############################################
# EDA ANALIZI
############################################

check_df(df)

# Bağımlı değişkende 59 tane NA var!

# BAĞIMLI DEĞİŞKEN ANALİZİ
df["Salary"].describe()
sns.distplot(df.Salary)
plt.show()

sns.boxplot(df["Salary"])
plt.show()

# KATEGORİK VE NUMERİK DEĞİŞKENLERİN SEÇİLMESİ
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# KATEGORİK DEĞİŞKEN ANALİZİ
rare_analyser(df, "Salary", cat_cols)

# SAYISAL DEĞİŞKEN ANALİZİ
for col in num_cols:
    num_summary(df, col, plot=False)

# AYKIRI GÖZLEM ANALİZİ
for col in num_cols:
    print(col, check_outlier(df, col))

# 1350 den sonraki değerleri veri setinden çıkartıyorum.
print(df.shape)
df = df[(df['Salary'] < 1350) | (df['Salary'].isnull())]  # Eksik değerleri de istiyoruz.
print(df.shape)

sns.boxplot(df["Salary"])
plt.show()

sns.distplot(df.Salary)
plt.show()

# AYKIRI DEĞERLERİ BASKILAMA
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# EKSİK GÖZLEM ANALİZİ
# Salary bağımlı değişkeninde 59 Eksik Gözlem bulunmakta. Bunları çıkartmak bir çözüm yolu olabilir.
missing_values_table(df)


# KORELASYON ANALİZİ
def target_correlation_matrix(dataframe, corr_th=0.5, target="Salary"):
    """
    Bağımlı değişken ile verilen threshold değerinin üzerindeki korelasyona sahip değişkenleri getirir.
    :param dataframe:
    :param corr_th: eşik değeri
    :param target:  bağımlı değişken ismi
    :return:
    """
    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("Yüksek threshold değeri, corr_th değerinizi düşürün!")


target_correlation_matrix(df, corr_th=0.5, target="Salary")

############################################
# VERİ ÖNİŞLEME
############################################

df['NEW_HitRatio'] = df['Hits'] / df['AtBat']
df['NEW_RunRatio'] = df['HmRun'] / df['Runs']
df['NEW_CHitRatio'] = df['CHits'] / df['CAtBat']
df['NEW_CRunRatio'] = df['CHmRun'] / df['CRuns']

df['NEW_Avg_AtBat'] = df['CAtBat'] / df['Years']
df['NEW_Avg_Hits'] = df['CHits'] / df['Years']
df['NEW_Avg_HmRun'] = df['CHmRun'] / df['Years']
df['NEW_Avg_Runs'] = df['CRuns'] / df['Years']
df['NEW_Avg_RBI'] = df['CRBI'] / df['Years']
df['NEW_Avg_Walks'] = df['CWalks'] / df['Years']


# One Hot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

############################################
# MODELLEME
############################################

# Salary içerisindeki boş değerleri ayıralım.
df_null = df[df["Salary"].isnull()]

# Salarydeki eksik değerleri çıkartma
df.dropna(inplace=True)

y = df['Salary']
X = df.drop("Salary", axis=1)


##########################
# BASE MODELS
##########################


def all_models(X, y, test_size=0.2, random_state=12345, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('GBM', GradientBoostingClassifier(random_state=random_state)),
                  ('XGB', XGBClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge(random_state=random_state)),
                  ("Lasso", Lasso(random_state=random_state)),
                  ("ElasticNet", ElasticNet(random_state=random_state)),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor(random_state=random_state)),
                  ('RF', RandomForestRegressor(random_state=random_state)),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor(random_state=random_state)),
                  ("XGBoost", XGBRegressor(random_state=random_state)),
                  ("LightGBM", LGBMRegressor(random_state=random_state)),
                  ("CatBoost", CatBoostRegressor(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return X_train, X_test, y_train, y_test, all_models_df

# Fonksiyonun Çalıştırılması
X_train, X_test, y_train, y_test, all_models = all_models(X, y, test_size=0.2, random_state=42, classification=False)

# 461
df["Salary"].mean()

##########################
# RANDOM FORESTS MODEL TUNING
##########################

# Tuning için hazırlanan parametreler. Tuning zaman aldığı için çıkan parametre değerlerini girdim.
# max_features: bir ağacı bölerken dikkate alınması gereken rastgele özellik alt kümelerinin boyutu.
# min_samples_split: bir ağacı bölmek için gereken minimum örnek sayısını temsil etmektedir.
rf_params = {"max_depth": [4, 5, 7, 10],
             "max_features": [4, 5, 6, 8, 10, 12],
             "n_estimators": [80, 100, 150, 250, 400, 500],
             "min_samples_split": [8, 10, 12, 15]}

best_params = {'max_depth': 10,
               'max_features': 5,
               'min_samples_split': 10,
               'n_estimators': 80}

# rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
# rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1).fit(X_train , y_train)
# **rf_cv_model.best_params_

# RANDOM FORESTS TUNED MODEL
rf_tuned = RandomForestRegressor(max_depth=10, max_features=5, n_estimators=80,
                                 min_samples_split=10, random_state=42).fit(X_train, y_train)

# TUNED MODEL TRAIN HATASI
y_pred = rf_tuned.predict(X_train)
print("RF Tuned Model Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred)))

##########################
# TUNED MODEL TEST HATASI
##########################

y_pred = rf_tuned.predict(X_test)
print("RF Tuned Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

##########################
# SALARYDEKİ NA DEĞERLERİN TAHMİNİ İLE BİRLİKTE MODEL KURMA
##########################

# NA Olan Salary Değerlerini kurduğumuz model ile test edip, yeni bir model kuralım.
df_null.head()
df_null.drop("Salary", axis=1, inplace=True)

salary_pred = rf_tuned.predict(df_null)

# tahmin edilen salary değerleri
salary_pred[0:5]

df_null['Salary'] = salary_pred

# yeni değişkenleri orjinal veri setine ekleme
df_final = pd.concat([df, df_null])

y = df_final["Salary"]
X = df_final.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Tuned parametleri aşağıda girildi. Girilmeseydi bu şekilde olacaktı.
# rf_final_model = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)
rf_final_model = RandomForestRegressor(max_depth=10, max_features=5, n_estimators=80, min_samples_split=10,
                                       random_state=42).fit(X_train, y_train)

# Train Hatası
y_pred_train = rf_final_model.predict(X_train)
print("TRAIN DATA TRAIN RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))

# Test Hatası
y_pred_test = rf_final_model.predict(X_test)
print("TEST DATA FINAL RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

#######################################
# FEATURE IMPORTANCE
#######################################


def plot_importance(model, features, num=len(X), save=False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final_model, X)