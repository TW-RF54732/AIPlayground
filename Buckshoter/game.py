import random
from config import setItem, setHealth

V1_props = ["magnifier","cigarette","beer","saw","handcuffs"]
V2_props = ["phone","Reverser","Epinephrine","medications"]

allowItems = V1_props

ACTION_SPACE = {
    0: "shoot_opponent",
    1: "shoot_self",
    2: "use_item_magnifier",
    3: "use_item_cigarette",
    4: "use_item_beer",
    5: "use_item_saw",
    6: "use_item_handcuffs"  
}

class GameEnv:
    def __init__(self):
        # 公共資訊
        self.turn = 0          # 當前行動者 (0=player1, 1=player2)
        self.round = 1         # 第幾回合
        self.bullets_live = 0  # 紅彈數量
        self.bullets_blank = 0 # 灰彈數量

        # 個別玩家資訊
        self.players = {
            "player1": {"health": 3, "items": [], "skip": False},
            "player2": {"health": 3, "items": [], "skip": False}
        }
        
        self.itemConfig = False

    def reset(self):
        """初始化遊戲"""
        self.turn = 0
        self.round = 1
        self.bullets_live = 0
        self.bullets_blank = 0
        self.players["player1"] = {"health": 3, "items": [], "skip": False}
        self.players["player2"] = {"health": 3, "items": [], "skip": False}
        self.ready = False
        self.allowTools = allowItems
        self.startUp()
        return self.get_state()

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
            "ready" : self.ready,
            "allowItems" : allowItems
        }

    def startUp(self, customHealth=False, Bullet=None, customItem=False):
        if Bullet is None:
            total = random.randint(3, 8)
            red = random.randint(1, total - 1)   # 至少 1
            gray = total - red
            Bullet = {"live": red, "blank": gray}
        
        self.bullets_blank = Bullet["blank"]
        self.bullets_live = Bullet["live"]

        if customItem:
            p1Item, p2Item = setItem(self.get_state())
            self.players["player1"]["items"] = p1Item
            self.players["player2"]["items"] = p2Item
        else:
            itemAmount = random.randint(0,8)
            self.players["player1"]["items"] = random.choices(allowItems,k=itemAmount)
            self.players["player2"]["items"] = random.choices(allowItems,k=itemAmount)

        if not customHealth:
            health = random.randint(3,6)
            self.players["player1"]["health"] = health
            self.players["player2"]["health"] = health
        else:
            p1hp, p2hp = setHealth(self.get_state())
            self.players["player1"]["health"] = p1hp
            self.players["player2"]["health"] = p2hp

    def step(self, action):
        """
        執行一個動作
        return: obs, reward, done, info
        """
        player = "player1" if self.turn == 0 else "player2"
        opponent = "player2" if self.turn == 0 else "player1"
        reward = 0
        done = False
        info = {}

        # 1. 執行動作
        if ACTION_SPACE[action] == "shoot_opponent":
            # 決定子彈
            if random.random() < self.bullets_live / max(1, (self.bullets_live + self.bullets_blank)):
                # 實彈
                self.players[opponent]["health"] -= 1
                self.bullets_live -= 1
                reward = 1  # 擊中對手
            else:
                self.bullets_blank -= 1
                reward = -0.1  # 空彈，沒造成傷害
            self.turn = 1 - self.turn  # 換人

        elif ACTION_SPACE[action] == "shoot_self":
            if random.random() < self.bullets_live / max(1, (self.bullets_live + self.bullets_blank)):
                # 實彈
                self.players[player]["health"] -= 1
                self.bullets_live -= 1
                reward = -1  # 打到自己
            else:
                self.bullets_blank -= 1
                reward = 0.5  # 空彈 = 證明下次安全
            self.turn = 1 - self.turn  # 換人

        else:
            # 之後可以加道具邏輯
            reward = -0.01

        # 2. 檢查遊戲是否結束
        if self.players[player]["health"] <= 0:
            done = True
            reward = -1
        if self.players[opponent]["health"] <= 0:
            done = True
            reward = 1

        obs = self.get_state()
        return obs, reward, done, info
