import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score

SALE_PRICE = 'SalePrice'

# ------------------------------------------------------------------------
# Подготовка данных
# ------------------------------------------------------------------------


# Разделить на категориальные и колличественные признаки
def spread(data):
    categorical_columns = \
        [col for col in data.columns if data[col].dtype.name == 'object']

    numerical_columns = \
        [col for col in data.columns if (col != SALE_PRICE and data[col].dtype.name != 'object')]

    price_column = \
        [col for col in data.columns if (col == SALE_PRICE)]

    return categorical_columns, numerical_columns, price_column


# Подготовить данные
def fillna(data, categorical_columns):

    # Заполним отсутсвующие значения колличественных признаков
    # медианными в текущем столбце
    data.fillna(data.median(axis=0), axis=0)
    # Заполнение отсутсвующих категориальных данных самыми популярными
    described_data = data.describe(include=[object])
    for col in categorical_columns:
        data[col] = data[col].fillna(described_data[col]['top'])


    return data, described_data


# Векторизация данных
def vectorize(data, described_data, categorical_cols, numerical_cols, price_column):
    # Признаки, принимающие бинарные значения
    binary_cols = \
        [col for col in categorical_cols if described_data[col]['unique'] == 2]
    # Признаки, принимающие небинарные значения
    nonbinary_cols = \
        [col for col in categorical_cols if described_data[col]['unique'] > 2]

    # Преобразуем данные бинарных признаков в 1 / 0
    for col in binary_cols:
        top = described_data[col]['top']
        top_items = data[col] == top
        data.loc[top_items, col] = 0
        data.loc[np.logical_not(top_items), col] = 1

    # Векторизация небинарных признаков
    data_nonbinary = pd.get_dummies(data[nonbinary_cols])

    # Нормализация количественных признаков
    data_numerical = data[numerical_cols]
    data_numerical = \
        (data_numerical - data_numerical.mean()) / data_numerical.std()
    data_price = data[price_column]

    # Слить в одну таблицу
    data = pd.concat((
        data_numerical,
        data[binary_cols],
        data_nonbinary,
        data_price
    ), axis=1)

    return pd.DataFrame(data, dtype=float)


# Подготовить данные
def prepare(data):
    categorical_cols, numerical_cols, price_column = spread(data)
    data, described_data = fillna(data, categorical_cols)
    data = vectorize(data, described_data, categorical_cols, numerical_cols, price_column)

    return data


# ------------------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------------------


train = pd.read_csv('./data/train.csv')
train = prepare(train)

# ------------------------------------------------------------------------
# Отбор информативных фич
# ------------------------------------------------------------------------


all_features = train.corr()[SALE_PRICE].abs()

actual_features = \
    [(f, all_features[f]) for f in train.columns if all_features[f] > 0.5]

actual_features = sorted(actual_features, key=lambda x: x[::-1], reverse=True)

# Отберем значимые фичи, т.к. знаем смысл каждой
MEANING_FEATURES = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'TotalBsmtSF',
    'ExterQual_TA',
    'FullBath',
    'BsmtQual_Ex',
    'TotRmsAbvGrd',
    'YearBuilt',
    'KitchenQual_TA',
    'GarageFinish_Unf',
    'KitchenQual_Ex',
]

train = train[MEANING_FEATURES + [SALE_PRICE]]

# Удаление выбросов
train = train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]

# Разделение на обучающую и тестовую выборку
X = train.drop((SALE_PRICE), axis=1)
y = train[SALE_PRICE]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=11)


regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)
y_test_predict = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
print("MAE: %.2f" % mean_absolute_error(y_test, y_test_predict))
print('Variance score: %.2f' % r2_score(y_test, y_test_predict))

X_check = pd.read_csv('./data/test.csv')
y_check = pd.read_csv('./data/sample_submission.csv')

X_check = prepare(X_check)
X_check = X_check[MEANING_FEATURES]
y_check = y_check[[SALE_PRICE]]

# print(y_check)

print(X_check.columns[X_check.isnull().values.any()].tolist())

y_check_predict = regr.predict(X_check)

# print('Coefficients: \n', regr.coef_)
# print("MAE: %.2f" % mean_absolute_error(y_check, y_check_predict))
# print('Variance score: %.2f' % r2_score(y_check, y_check_predict))
