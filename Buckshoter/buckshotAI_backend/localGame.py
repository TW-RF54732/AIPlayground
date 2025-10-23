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
@router.post("/api/game/initGame")
def initGame(setting:gameSettings):
    global current_game
    current_game = GameEnv()
    current_game.reset()
    if setting.custombullet == None:
        current_game.customBullet = False
        print("Running Default bullet")
    else:
        current_game.customBullet = True
        current_game.bulletsList = setting.custombullet
        current_game.bullets_live = setting.custombullet.count(True)
        current_game.bullets_blank = setting.custombullet.count(False)

    if setting.customHealth == None:
        current_game.customHealth = False
        print("Running Default Health")
    else:
        current_game.customHealth = True
        current_game.players[0]["health"] = setting.customHealth
        current_game.players[1]["health"] = setting.customHealth

    if setting.customItem == None:
        current_game.customItem = False
        print("Running Default Item")
    else:
        current_game.customItem = True
        current_game.players[0]["items"] = setting.customItem[0]
        current_game.players[1]["items"] = setting.customItem[1]
    
    current_game.startUp()
    return JSONResponse(status_code=200,content={"message":f"Game started with {'custom' if current_game.customBullet == True else 'default'} bullte, {'custom' if current_game.customHealth == True else 'default'} health, {'custom' if current_game.customItem == True else 'default'} item"})

@router.get("/api/game/endGame")
def endGame():
    global current_game
    current_game = None
    return JSONResponse(status_code=200,content={"message":"Game closed"})

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
