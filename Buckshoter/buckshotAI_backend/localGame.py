from fastapi import APIRouter, Request,FastAPI
from mainGame.game import GameEnv

router = FastAPI()
current_game = None

'''
This is the place to put API
All API in game's route is /api/game/...
'''

@router.get("/api/game/initGame")
def initGame(request:Request):
    current_game = GameEnv()
    current_game.reset()
    return current_game.get_state()
