from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()


# ===== Todo App Models =====
class TodoCreate(BaseModel):
    title: str
    description: Optional[str] = None
    completed: bool = False


class TodoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    completed: Optional[bool] = None


class Todo(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    completed: bool = False


# In-memory storage for todos
todos_db: dict[int, Todo] = {}
todo_id_counter = 1

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


# ===== Todo API Endpoints =====
@app.get("/todos", response_model=List[Todo])
def get_todos():
    """Get all todos"""
    return list(todos_db.values())


@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    """Get a specific todo by ID"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todos_db[todo_id]


@app.post("/todos", response_model=Todo, status_code=201)
def create_todo(todo: TodoCreate):
    """Create a new todo"""
    global todo_id_counter
    new_todo = Todo(
        id=todo_id_counter,
        title=todo.title,
        description=todo.description,
        completed=todo.completed
    )
    todos_db[todo_id_counter] = new_todo
    todo_id_counter += 1
    return new_todo


@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, todo: TodoUpdate):
    """Update an existing todo"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")

    existing_todo = todos_db[todo_id]
    update_data = todo.dict(exclude_unset=True)

    updated_todo = Todo(
        id=existing_todo.id,
        title=update_data.get("title", existing_todo.title),
        description=update_data.get("description", existing_todo.description),
        completed=update_data.get("completed", existing_todo.completed)
    )
    todos_db[todo_id] = updated_todo
    return updated_todo


@app.delete("/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int):
    """Delete a todo"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    del todos_db[todo_id]
    return None
