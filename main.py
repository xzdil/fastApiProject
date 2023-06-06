from fastapi import FastAPI
from fastapi import Request, Depends
from fastapi.templating import Jinja2Templates
import uvicorn
from model import getPredict_cached

app = FastAPI()
templates = Jinja2Templates(directory="templates")

if __name__ == '__main__':
    uvicorn.run("main:app", host='localhost', port=8080, reload=True, workers=3)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict/")
async def predict(sentence: str):
    return await getPredict_cached(sentence)
