from fastapi import APIRouter, Request,FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from mainGame.game import GameEnv
router = FastAPI()
current_game = None

class gameSettings(BaseModel):
    customHealth: Optional[int] = None
    custombullet: Optional[list[bool]] = None
    customItem: Optional[list[list[str]]] = None

'''
This is the place to put API 
All API in game's route is /api/game/...
'''
@router.get("/api/game/startDefaultGame")
def startDefaultGame():
    global current_game
    current_game = GameEnv()
    current_game.reset()
    return JSONResponse(status_code=200,content={"message":"Game start!"})

@router.post("/api/game/initGame")
def initGame(setting:gameSettings):
    global current_game
    current_game = GameEnv()
    current_game.reset()
    if setting.custombullet == None:
        print("Running Default bullet")
        current_game.bulletsList = setting.custombullet
        current_game.customBullet = True
        current_game.bullets_blank = setting.custombullet.count

    if setting.customHealth == None:
        print("Running Default bullet")
        current_game.bulletsList = setting.custombullet
        current_game.customBullet = True
    if setting.customItem == None:
        print("Running Default bullet")
        current_game.bulletsList = setting.custombullet
        current_game.customBullet = True
        
    return current_game.get_state()

@router.get("/api/game/endGame")
def endGame():
    global current_game
    current_game = None
    return "Game closed"

'''
Game API
'''
@router.get("/api/game/getStatus")
def getStatus():
    global current_game
    return current_game.get_state()

@router.get("/api/game/getAllowItems")
def getAllowItems():
    if current_game:
        return current_game.allowTools
    else:
        return JSONResponse(status_code=409, content={"message": "You have to run the game before checking allow items"})