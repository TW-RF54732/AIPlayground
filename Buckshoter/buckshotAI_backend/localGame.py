from fastapi import FastAPI
import sys,os
current_dir = os.path.dirname(__file__)
# 專案根目錄：buckshotAI_backend 的上層
project_root = os.path.abspath(os.path.join(current_dir, '..'))
print(current_dir)
# 加入專案根目錄到 Python 模組搜尋路徑
if project_root not in sys.path:
    sys.path.append(project_root)

# 現在可以匯入 mainGame 內的模組
from mainGame.game import GameEnv


'''
This is the place to put API
All API in game's route is /api/game/...
'''
@app.get("/api/initGame")
def initGame():
    game.reset()
    return game.get_state()