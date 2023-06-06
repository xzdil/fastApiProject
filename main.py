from fastapi import FastAPI
from fastapi import Request, Depends
from fastapi.templating import Jinja2Templates
from model import getPredict_cached

app = FastAPI()
templates = Jinja2Templates(directory="templates")

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict/{encodedSentence}")
async def predict(encodedSentence: str):
    print(encodedSentence)
    return await getPredict_cached(encodedSentence)
