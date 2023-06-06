from fastapi import FastAPI
from fastapi import Request, Depends
from fastapi.templating import Jinja2Templates
import uvicorn
from model import getPredict_cached
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

if __name__ == '__main__':
    uvicorn.run("main:app", host='20.105.232.27', port=int(os.environ.get("PORT", 443)), reload=True, workers=3)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict/{encodedSentence}")
async def predict(encodedSentence: str):
    print(encodedSentence)
    return await getPredict_cached(encodedSentence)
