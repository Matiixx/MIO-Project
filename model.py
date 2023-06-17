import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def create_custom_model(data_path):
    data = pd.read_csv(f'{data_path}')
    data['content'] = data['content'].fillna('')

    X = data['content']
    y = data['favorites']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=321)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
    X_test_vectorized = vectorizer.transform(X_test).toarray()

    label_to_index = {label: index for index, label in enumerate(np.unique(y))}
    y_train = np.array([label_to_index[label] for label in y_train])
    y_test = np.array([label_to_index[label] for label in y_test])

    model = LinearRegression()
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


def predict_populatiry(post):
    pass
