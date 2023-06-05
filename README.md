# Решение набора данных "Титаник" с применением машинного обучения

Цель данного анализа - спроектировать модель машинного обучения, которая будет предсказывать по определённым параметрам,выжил человек на титанике во время кораблекрушения или нет.

### Первичный анализ данных

Сначала прочитаем датасет и проведём его первичный анализ:
```python
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("datasets/train.csv")
test_data = pd.read_csv("datasets/test.csv")

print(train_data.head(5))
print(train_data.describe())
```
Вывод:

| PassengerId | Survived | Pclass | Name                         | Sex    | Age | SibSp | Parch | Ticket           | Fare  | Cabin | Embarked |
|-------------|----------|--------|------------------------------|--------|-----|-------|-------|------------------|-------|-------|----------|
| 1           | 0        | 3      | Braund, Mr. Owen Harris      | male   | 22  | 1     | 0     | A/5 21171        | 7.25  |       | S        |
| 2           | 1        | 1      | Cumings, Mrs. John Bradley   | female | 38  | 1     | 0     | PC 17599         | 71.28 |       | C        |
| 3           | 1        | 3      | Heikkinen, Miss. Laina       | female | 26  | 0     | 0     | STON/O2. 3101282 | 7.93  |       | S        |
| 4           | 1        | 1      | Futrelle, Mrs. Jacques Heath | female | 35  | 1     | 0     | 113803           | 53.1  |       | S        |
| 5           | 0        | 3      | Allen, Mr. William Henry     | male   | 35  | 0     | 0     | 373450           | 8.05  |       | S        |

Описание переменных:

|       | PassengerId | Survived | PClass | Age    | SibSp | Parch | Fare   |
|-------|-------------|----------|--------|--------|-------|-------|--------|
| count | 891         | 891      | 891    | 714    | 891   | 891   | 891    |
| mean  | 446         | 0.(38)   | 2.31   | 29.7   | 0.52  | 0.38  | 32.21  |
| std   | 257.35      | 0.49     | 0.84   | 14.51  | 1.1   | 0.81  | 49.69  |
| min   | 1           | 0        | 1      | 0.42   | 0     | 0     | 0      |
| 25%   | 223.5       | 0        | 2      | 20.125 | 0     | 0     | 7.91   |
| 50%   | 446         | 0        | 3      | 28     | 0     | 0     | 14.45  |
| 75%   | 668.5       | 1        | 3      | 38     | 1     | 0     | 31     |
| max   | 891         | 1        | 3      | 80     | 8     | 6     | 512.33 |

Сразу можно заметить, что кол-во значений "Age" меньше 891, значит у нас есть пустые значения, которые мы отбросим в тренировочном наборе, а в тестовом заменим медианами. Также нам следует разбить возраст на 6 классов и факторизовать переменную пола (заменить категории значениями 0 и 1).

```python
train_data = train_data.dropna(subset = ["Age"])  ### Отброс неизвестных значений возраста

age_median = test_data["Age"].median()
test_data["Age"].fillna(age_median, inplace = True)   ### Замена n/a возраста медианами в тестовом наборе

fare_median = test_data["Fare"].median()
test_data["Fare"].fillna(fare_median, inplace = True)   ### Замена n/a fare в тестовом наборе
```

Разбиение "Age" на классы и факторизация пола:

```python
train_data["Age_cat"] = np.ceil(train_data["Age"] / 15)  ### Классификация возрастов на 6 классов
test_data["Age_cat"] = np.ceil(test_data["Age"] / 15)

train_data_cat = train_data["Sex"]
train_data_cat_encoded, train_data_cat_categories = train_data_cat.factorize()  ###  Факторизация категориальной переменной пола

test_data_cat = test_data["Sex"]
test_data_cat_encoded, test_data_cat_categories = test_data_cat.factorize()

train_data["Sex"] = train_data_cat_encoded
test_data["Sex"] = test_data_cat_encoded
```

### Определение зависимостей и подготовка данных для обучения

Выведем корреляционную матрицу для значений набора данных:

```python
corr_matrix = train_data.corr(numeric_only = True)
```

|             | PassengerId | Survived | Pclass | Sex    | Age    | SibSp  | Parch  | Fare   | Age_cat |
|-------------|-------------|----------|--------|--------|--------|--------|--------|--------|---------|
| PassengerId | 1           | 0.029    | -0.035 | -0.024 | 0.036  | -0.082 | -0.011 | 0.009  | 0.054   |
| Survived    | 0.029       | 1        | -0.359 | 0.538  | -0.077 | -0.017 | 0.093  | 0.268  | -0.067  |
| Pclass      | -0.035      | -0.359   | 1      | -0.155 | -0.369 | 0.067  | 0.025  | -0.554 | -0.370  |
| ...         | ...         | ...      | ...    | ...    | ...    | ...    | ...    | ...    | ...     |
| Age_cat     | 0.054       | -0.067   | -0.370 | -0.104 | 0.954  | -0.287 | -0.180 | 0.101  | 1       |

Замечаем, что лучше всего с атрибутом "Survived" коррелируют такие значения, как "Pclass", "Sex" и "Fare". Следовательно на них мы и будем опираться при обучении модели. Подготовим данные для модели.

```python
train_data_prepared = train_data[["Pclass", "Sex", "Fare"]].copy()
train_data_labels = train_data["Survived"].copy()

test_data_prepared = test_data[["Pclass, "Sex", "Fare"]].copy()
gender_submission = pd.read_csv("datasets/gender_submission.csv)  ### Файл, содержащий лейблы тестового датасета для дальнейшей оценки эффективности модели
test_data_labels = gender_submission["Survived"]
```

### Обучение модели и оценка эффективности

Для решения данной задачи подойдет алгоритм классификации случайного леса. Создадим модель, настроим ее под данные с помощью решетчатого поиска и обучим ее:

```python
from sklearn_ensemble import RandomForestClassifier

model = RandomForestClassifier()

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grind_search = RandomizedSearchCV(model, param_grid, cv = 5, scoring = "neg_mean_squared_error")
grind_search.fit(train_data_prepared, train_data_labels)

model = grind_search.best_estimator_

model.fit(train_data_prepared, train_data_labels)
```

Теперь заставим модель предсказать значения на обучающем датасете и рассчитаем эффективность модели.

```python
from sklearn.metric import mean_squared_error

predictions = model.predict(train_data_prepared)

model_mse = mean_squared_error(train_data_labels, predictions)
model_rmse = np_sqrt(model_mse)

print(model_rmse) ### 0.32
```

Корень из среднеквадратической ошибки приблизительно равен 0.32, что является неплохим результатом для такой просто модели.

### Запуск модели и составление прогнозов

Теперь можем сгенерировать предсказания для нашего тестового набора данных и затем рассчитать, насколько хорошо наша модель обобщается на новые данные:

```python
test_predictions = pd.DataFrame(np.array(model.predict(test_data_prepared)).round().astype("int"))

test_mse = mean_squared_error(test_data_labels, test_predictions)
test_rmse = np.sqrt(model_mse)

print(test_rmse)  ### 0.32...

filename = "titanic_predictions.csv"

test_predictions.to_csv(f"datasets/{filename}", index = False)  ### Сохранение прогнозов в отдельный файл.
```

Посчитав в очередной раз корень из среднеквадратической ошибки получаем, что наша модель довольно хорошо обобщается на новые данные.

Источник наборов данных - [ссылка](https://www.kaggle.com/competitions/titanic/data)
