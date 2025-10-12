from fastapi import FastAPI

app = FastAPI()

@app.get("/Game")
async def root():
    return "Hello" 