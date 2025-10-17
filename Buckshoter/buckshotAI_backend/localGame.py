from fastapi import APIRouter, Request,FastAPI
from pydantic import BaseModel

from mainGame.game import GameEnv
router = FastAPI()
current_game = None

class gameSettings(BaseModel):
    customHealth: bool
    custombullet: bool
    customItem: bool

'''
This is the place to put API
All API in game's route is /api/game/...
'''

@router.post("/api/game/initGame")
def initGame(setting:gameSettings):
    global current_game
    current_game = GameEnv()
    current_game.reset()
    return current_game.get_state()

@router.get("/api/game/endGame")
def endGame():
    global current_game
    current_game = None
    return "Game closed"

'''
Game API
'''
@router.get("/api/game/getObs")
def getObs():
    global current_game
    return current_game.get_state()