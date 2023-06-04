import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv("titanic_ds/datasets/train.csv")
test_data = pd.read_csv("titanic_ds/datasets/test.csv")
gender_submission = pd.read_csv("titanic_ds/datasets/gender_submission.csv")

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

model = RandomForestRegressor()


train_data_prepared = train_data[["Pclass", "Sex", "Fare"]].copy()
train_data_labels = train_data["Survived"].copy()               ### Подготовленные данные для обучения модели


###  Тренировка модели на обучающем наборе данных и её оценка
model.fit(train_data_prepared, train_data_labels)

predictions = model.predict(train_data_prepared)


model_mse = mean_squared_error(train_data_labels, predictions)
model_rmse = np.sqrt(model_mse)

test_data_prepared = test_data[["Pclass", "Sex", "Fare"]].copy()

test_predictions = pd.DataFrame(np.array(model.predict(test_data_prepared)).round().astype("int"))            ###  Предикты на тренировочном наборе данных

test_data_labels = gender_submission["Survived"]

test_mse = mean_squared_error(test_data_labels, test_predictions)   ### Ошибка модели на тестовом наборе данных
test_rmse = np.sqrt(model_mse)

print(test_rmse)

filename = "titanic_predictions.csv"

test_predictions.to_csv(f"titanic_ds/datasets/{filename}", index = False)    ### Сохранение предиктов в файл 'titanic_predictions.csv'

