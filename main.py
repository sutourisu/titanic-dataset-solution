import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv("titanic_ds/datasets/train.csv")
test_data = pd.read_csv("titanic_ds/datasets/test.csv")

print(train_data.describe())    ### Общая характеристика данных

train_data = train_data.dropna(subset = ["Age"])        ###  Дроп неизвестных возрастов

age_median = test_data["Age"].median()
test_data["Age"].fillna(age_median, inplace = True)

fare_median = test_data["Fare"].median()
test_data["Fare"].fillna(fare_median, inplace = True)   ###  Заполнение неизвестных полей медианами в тестовом наборе данных

'''Нужно факторизовать переменную пола, а также классифицировать возраст на более крупные классы'''

train_data["Age_cat"] = np.ceil(train_data["Age"] / 15)  ### Классификация возрастов на 6 классов
test_data["Age_cat"] = np.ceil(test_data["Age"] / 15)

train_data_cat = train_data["Sex"]
train_data_cat_encoded, train_data_cat_categories = train_data_cat.factorize()  ###  Факторизация категориальной переменной пола

test_data_cat = test_data["Sex"]
test_data_cat_encoded, test_data_cat_categories = test_data_cat.factorize()

train_data["Sex"] = train_data_cat_encoded
test_data["Sex"] = test_data_cat_encoded

corr_matrix = train_data.corr(numeric_only = True)      ###  Корреляционная матрица значений

print(corr_matrix)

#  Наиболее перспективные для отбора атрибуты: "Pclass", "Sex", "Fare".

split_train, split_test = train_test_split(train_data, test_size = 0.2, random_state = 42)

train_data_prepared = split_train[["Pclass", "Sex", "Fare"]].copy()
train_data_labels = split_train["Survived"].copy()               ### Подготовленные данные для обучения модели

model = RandomForestClassifier()

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grind_search = RandomizedSearchCV(model, param_grid, cv = 5, scoring = "neg_mean_squared_error")
grind_search.fit(train_data_prepared, train_data_labels)

model = grind_search.best_estimator_

###  Тренировка модели на обучающем наборе данных и её оценка
model.fit(train_data_prepared, train_data_labels)

print(cross_val_score(model, train_data_prepared, train_data_labels, cv = 3))

test_data_prepared = split_test[["Pclass", "Sex", "Fare"]].copy()

test_predictions = np.array(model.predict(test_data_prepared))          ###  Предикты на тренировочном наборе данных

test_data_labels = split_test["Survived"]

print(cross_val_score(model, test_data_prepared, test_data_labels, cv = 3))

ids = test_data["PassengerId"].copy()
final_data_prepared = test_data[["Pclass", "Sex", "Fare"]].copy()

final_predictions = model.predict(final_data_prepared)        ### Финальные предикты

file = pd.DataFrame({"PassengerId": ids, "Survived": final_predictions})
file.to_csv("titanic_ds/datasets/final_submission.csv", index = False)