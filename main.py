from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

clf = pickle.load(open("model.pickle", "rb"))


class Predict(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: str


def _titanic_predict(df):
    df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    df['Age'].fillna(999, inplace=True)
    df['Embarked'].fillna(('S'), inplace=True)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    delete_columns = ['Name', 'PassengerId', 'SibSp',
                      'Parch', 'Ticket', 'Cabin', 'Fare']
    df.drop(delete_columns, axis=1, inplace=True)

    return clf.predict_proba(df)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/titanic/")
def titanic(req: Predict):
    df = pd.DataFrame([req.dict()])
    prediction = _titanic_predict(df)
    return {"res": "ok", "proba of survive": prediction[0][1]}
