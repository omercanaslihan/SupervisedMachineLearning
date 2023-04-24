##Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma##

##İş Problemi##
#Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
#(average, highlighted) oyuncu olduğunu tahminleme

#Veri Seti Hikayesi#
#Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
#içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

#Görev1
#Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
att_df = pd.read_csv("scoutium_attributes.csv", sep=";")
pot_df = pd.read_csv("scoutium_potential_labels.csv", sep=";")
att_df.head()
pot_df.head()
#Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
#("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
df = att_df.merge(pot_df, on=["task_response_id", "match_id", "evaluator_id", "player_id"])
df.head()
#Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
df = df.loc[~(df["position_id"] == 1)]
df.head()
#Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
df = df.loc[~(df["potential_label"] == "below_average")]
df.head()
#Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
#olacak şekilde manipülasyon yapınız.
#Görevler
#Adım 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
#“attribute_value” olacak şekilde pivot table’ı oluşturunuz
pivot_df = pd.pivot_table(df, values="attribute_value", index=["player_id", "position_id", "potential_label"], columns=["attribute_id"])
pivot_df.head()
#Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
pivot_df = pivot_df.reset_index()
pivot_df.head()
pivot_df.columns = pivot_df.columns.astype("str")
#Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
le = LabelEncoder()
pivot_df["potential_label"] = le.fit_transform(pivot_df["potential_label"])
pivot_df.head()
#Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
num_cols = [col for col in pivot_df.columns if pivot_df[col].dtypes != "O" and pivot_df[col].nunique() > 5]
num_cols = [col for col in num_cols if col not in "player_id"]
num_cols = num_cols[1:]
#Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
ss = StandardScaler()
df = pivot_df.copy()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()
#Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
#geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
X = df.drop(["potential_label"], axis=1)
y = df["potential_label"]

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    models = [('LR', LogisticRegression()),
          ("KNN", KNeighborsClassifier()),
          ("CART", DecisionTreeClassifier()),
          ("RF", RandomForestClassifier()),
          ('GBM', GradientBoostingClassifier()),
          ("XGBoost", XGBClassifier(eval_metric='logloss')),
          ("LightGBM", LGBMClassifier()),
          ("CatBoost", CatBoostClassifier(verbose=False))]

    for name, classifier in models:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y)
#roc_auc: 0.5551 (LR)
#roc_auc: 0.5 (KNN)
#roc_auc: 0.7105 (CART)
#roc_auc: 0.8972 (RF)
#roc_auc: 0.8177 (GBM)
#roc_auc: 0.8308 (XGBoost)
#roc_auc: 0.8493 (LightGBM)
#roc_auc: 0.8887 (CatBoost)
base_models(X, y, scoring="f1")
#f1: 0.0 (LR)
#f1: 0.184 (KNN)
#f1: 0.3837 (CART)
#f1: 0.5834 (RF)
#f1: 0.4818 (GBM)
#f1: 0.548 (XGBoost)
#f1: 0.5586 (LightGBM)
#f1: 0.6164 (CatBoost)
base_models(X, y, scoring="accuracy")
#accuracy: 0.7934 (LR)
#accuracy: 0.5547 (KNN)
#accuracy: 0.8081 (CART)
#accuracy: 0.856 (RF)
#accuracy: 0.7571 (GBM)
#accuracy: 0.8193 (XGBoost)
#accuracy: 0.8268 (LightGBM)
#accuracy: 0.8744 (CatBoost)
base_models(X, y, scoring="precision")
#precision: 0.1318 (KNN)
#precision: 0.4326 (CART)
#precision: 0.7148 (RF)
#precision: 0.5671 (GBM)
#precision: 0.5839 (XGBoost)
#precision: 0.6704 (LightGBM)
#precision: 0.8259 (CatBoost)

catboost_model = CatBoostClassifier(random_state=17, verbose=False)
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
catboost_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=False).fit(X, y)
catboost_final = catboost_model.set_params(**catboost_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision"])

cv_results['test_accuracy'].mean() #0.8781144781144782
cv_results['test_f1'].mean() #0.5842366712571316
cv_results['test_roc_auc'].mean() #0.8866102889358703
cv_results['test_precision'].mean() #0.9236363636363636

#Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
def plot_importance(model, features, num=len(X), save=False):
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

plot_importance(catboost_final, X)