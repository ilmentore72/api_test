from typing import Optional,List
import uvicorn
from fastapi import FastAPI, Response, Request , Body
from pydantic import BaseModel
class point(BaseModel):
    lat: str
    long: str
class point_arr(BaseModel):
    list : List[point] = []

class data_res(BaseModel):
    UID:str
    message:str
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello"}

@app.post("/data")
def return_UID(data:point_arr): 
    print(data)
    return data

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run(app,host =  "0.0.0.0", port = 8000)