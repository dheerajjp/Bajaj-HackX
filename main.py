from fastapi import FastAPI, UploadFile, Form
from routes import ingest, query

app = FastAPI()

app.include_router(ingest.router, prefix="/ingest")
app.include_router(query.router, prefix="/query")
