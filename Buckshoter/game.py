import random
V1_props = ["magnifier","cigarette","beer","saw","handcuffs"]
V2_props = ["phone","Reverser","Epinephrine","medications"]

allowItems = V1_props

class GameEnv:
    def __init__(self):
        # 公共資訊
        self.turn = 0          # 當前行動者 (0=player1, 1=player2)
        self.round = 1         # 第幾回合
        self.bullets_live = 0  # 紅彈數量
        self.bullets_blank = 0 # 灰彈數量

        # 個別玩家資訊
        self.players = {
            "player1": {
                "health": 3,
                "items": []
            },
            "player2": {
                "health": 3,
                "items": []
            }
        }
        
        self.itemConfig = False

    def reset(self):
        """初始化遊戲"""
        self.turn = 0
        self.round = 1
        self.bullets_live = 0
        self.bullets_blank = 0
        self.players["player1"] = {"health": 3, "items": []}
        self.players["player2"] = {"health": 3, "items": []}
        self.ready = False
        self.allowToos = allowItems

    def get_state(self):
        """把當前遊戲狀態轉成 dict(方便丟給AI觀察用)"""
        return {
            "turn": self.turn,
            "round": self.round,
            "bulletsOnTable": {
                "live": self.bullets_live,
                "blank": self.bullets_blank
            },
            "players": {
                "player1": self.players["player1"].copy(),
                "player2": self.players["player2"].copy()
            },
            "ready" : False,
            "allowItems" : allowItems
            
        }

    def startUp(self,Bullet = None,customItem = False):
        if(Bullet==None):
            total = random.randint(3, 8)
            # red 至少 1，gray 至少 1
            red = random.randint(1, total - 1)
            gray = total - red
            Bullet = {"live": red, "blank": gray}
        
        self.bullets_blank = Bullet["blank"]
        self.bullets_live = Bullet["live"]
        if customItem:
            from config import setItem
            setItem(self.get_state())
        else:
            itemAmount = random.randint(0,8)
            self.players["player1"]["items"] = random.choices(allowItems,k=itemAmount)
            self.players["player2"]["items"] = random.choices(allowItems,k=itemAmount)
    
        

