from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle


def create_custom_model(data_path):
    data = pd.read_csv(f"{data_path}")
    data["content"] = data["content"].fillna("")

    X = data["content"]
    y = data["favorites"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=321
    )

    vectorizer = TfidfVectorizer(min_df=20)
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
    X_test_vectorized = vectorizer.transform(X_test).toarray()

    label_to_index = {label: index for index, label in enumerate(np.unique(y))}
    y_train = np.array([label_to_index[label] for label in y_train])
    y_test = np.array([label_to_index[label] for label in y_test])

    model = xgb.XGBRegressor()
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)
    y_train_pred = model.predict(X_train_vectorized)

    mse = mean_squared_error(y_train, y_train_pred)
    print("Mean Squared Error:", mse)

    mae = mean_absolute_error(y_train, y_train_pred)
    print("Mean Absolute Error:", mae)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    filename = "finalized_model.pkl"
    pickle.dump(model, open(filename, "wb"))

    filename = "vectorizer.pkl"
    pickle.dump(vectorizer, open(filename, "wb"))


def predict_populatiry(content):
    filename = "finalized_model.pkl"
    model = pickle.load(open(filename, "rb"))
    filename = "vectorizer.pkl"
    vectorizer = pickle.load(open(filename, "rb"))
    content_vectorized = vectorizer.transform([content]).toarray()
    return model.predict(content_vectorized)[0]
