from typing import Optional
import uvicorn
from fastapi import FastAPI, Response, Request , Body
from pydantic import BaseModel
class data_req(BaseModel):
    UID:str
class data_res(BaseModel):
    UID:str
    message:str
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello"}

@app.post("/data")
def return_UID(data:data_req): 
    print (data)   
    return{"UID":data.UID, "Message":"POST Success"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run(app,host =  "0.0.0.0", port = 8000)