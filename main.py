import pandas as pd
import re
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

# предобработки текста
def preprocess_text(text):
    text = str(text).lower()  # Приведение к строковому типу и нижнему регистру
    text = re.sub('[^a-zA-Z0-9]', ' ', text)  # Замена нежелательных символов на пробелы
    return text

# загрузка данных и их предобработка
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['ProcessedDescription'] = data['FullDescription'].apply(preprocess_text)
    data['LocationNormalized'] = data['LocationNormalzed'].fillna('nan')
    data['ContractTime'] = data['ContractTime'].fillna('nan')
    return data

# векторизации текстовых данных с использованием TF-IDF
def vectorize_text(data):
    vectorizer = TfidfVectorizer(min_df=5)
    X_text = vectorizer.fit_transform(data['ProcessedDescription'])
    return X_text, vectorizer

# векторизации категориальных данных с использованием one-hot кодировавния
def vectorize_categorical(data):
    categorical_columns = ['LocationNormalized', 'ContractTime']
    categorical_data = data[categorical_columns].to_dict(orient='records')
    dict_vectorizer = DictVectorizer(sparse=True)
    X_categorical = dict_vectorizer.fit_transform(categorical_data)
    return X_categorical, dict_vectorizer

# объединения текстовых и категориальных признаков в единую матрицу признаков
def combine_features(X_text, X_categorical):
    X = hstack([X_text, X_categorical])
    return X


# функция для разделения данных на обучающую и тестовую выборки
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# функция для обучения модели гребневой регрессии (Ridge)
def train_ridge_model(X_train, y_train, alpha=1.0):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    return ridge_model

def main():
    # Шаг 1: Загрузка и предобработка данных
    data = load_and_preprocess_data('salary-train.csv')
    print("Первые 5 строк данных:")
    print(data.head())

    # Шаг 2: Преобразование текста в признаки TF-IDF
    X_text, text_vectorizer = vectorize_text(data)
    print(f"\nРазмер TF-IDF матрицы: {X_text.shape}")

    # Шаг 3: Преобразование категориальных признаков
    X_categorical, dict_vectorizer = vectorize_categorical(data)
    print(f"Размер матрицы после one-hot-кодирования: {X_categorical.shape}")

    # Шаг 4: Объединение всех признаков
    X = combine_features(X_text, X_categorical)
    print(f"Размер объединенной матрицы признаков: {X.shape}")

    # Шаг 5: Целевая переменная
    y = data['SalaryNormalized']

    # Шаг 6: Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nРазмер обучающей выборки: {X_train.shape[0]} примеров")
    print(f"Размер тестовой выборки: {X_test.shape[0]} примеров")

    # Шаг 7: Обучение модели гребневой регрессии
    model = train_ridge_model(X_train, y_train)
    print("\nМодель гребневой регрессии успешно обучена.")

    # Шаг 8: Предсказание на тестовой выборке
    y_pred = model.predict(X_test)
    print("\nПредсказания для тестовой выборки:", y_pred)


if __name__ == '__main__':
    main()