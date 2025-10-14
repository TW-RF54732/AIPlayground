from fastapi import FastAPI
from mainGame.game import GameEnv

app = FastAPI()
game = GameEnv()

'''
This is the place to put API
All API in game's route is /api/game/...
'''

@app.get("/api/initGame")
def initGame():
    game.reset()
    return game.get_state()