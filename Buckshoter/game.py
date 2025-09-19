import random
from config import setItem,setHealth
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
            0: {
                "health": 3,
                "items": [],
                "skip": False
            },
            1: {
                "health": 3,
                "items": [],
                "skip": False
            }
        }
        
        self.itemConfig = False

    def reset(self):
        """初始化遊戲"""
        # game data
        self.turn = 0
        self.round = 1
        self.maxHealth = 0
        self.itemPerRound = 0
        self.bullets_live = 0
        self.bullets_blank = 0
        self.bulletsList = []
        self.players[0] = {"health": 0, "items": [], "skip" : False}
        self.players[1] = {"health": 0, "items": [], "skip" : False}
        self.shortened = False
        self.magnifier_result = None
        # env setting
        self.allowToos = allowItems
        self.startUp()

    def get_state(self):
        """把當前遊戲狀態轉成 dict(方便丟給AI觀察用)"""
        return {
            "turn": self.turn,
            "round": self.round,
            "maxHealth":self.maxHealth,
            "itemPerRound": self.itemPerRound,
            "bulletsOnTable": {
                "live": self.bullets_live,
                "blank": self.bullets_blank
            },
            "players": {
                0: self.players[0].copy(),
                1: self.players[1].copy()
            },
            "shorted": self.shortened,
            "magnifier_result": self.magnifier_result,
            "action_mask" : self.get_action_mask()
        }

    def startUp(self,customHealth = False,Bullet = None,customItem = False):
        if(Bullet==None):
            total = random.randint(3, 8)
            # red 至少 1，gray 至少 1
            red = random.randint(1, total - 1)
            gray = total - red
            Bullet = {"live": red, "blank": gray}
            self.bulletsList = random.sample([True]*(Bullet["live"]) + [False]*(Bullet["blank"]), total)
        
        self.bullets_blank = Bullet["blank"]
        self.bullets_live = Bullet["live"]
        if customItem:
            
            p1Item, p2Item = setItem(self.get_state())
            self.players[0]["items"] = p1Item
            self.players[1]["items"] = p2Item

        else:
            self.itemPerRound = random.randint(0,8)
            self.players[0]["items"] = self.getRandItem(self.itemPerRound)
            self.players[1]["items"] = self.getRandItem(self.itemPerRound)

        if customHealth == False:
            health = random.randint(3,6)
            self.maxHealth = health
            self.players[0]["health"] = health
            self.players[1]["health"] = health
        else:
            p1hp,p2hp,maxHp = setHealth(self.get_state())
            self.players[0]["health"] = p1hp
            self.players[1]["health"] = p2hp
            self.maxHealth = maxHp
    
    def step(self, action):
        """
        執行一個動作
        return: obs, reward, done, info
        """
        player = self.turn
        opponent = 1 - player
        reward = 0
        done = False
        info = {}
        # if  self.players[player]["skip"]==True:
        #     self.turn = 1 - self.turn
        #     self.players[player]["skip"]=False
        #     obs = self.get_state()
        #     # print(f"Player: {player} is skipped!")
        #     return obs, reward,done,info


        # 先檢查：還有沒有子彈
        if not self.bulletsList:
            self.newRound()
            self.round +=1
            info["round"] = self.round
            # print(f"New Round, bullet list: {self.bulletsList}")
            return self.get_state(), reward, done, info
        # 1. 執行動作
        if ACTION_SPACE[action] == "shoot_opponent":
            current_bullet = self.bulletsList.pop(0)
            # print(f"Origin: Live: {self.bullets_live}, Blank: {self.bullets_blank}.\n Opponent: {opponent},health:{self.players[opponent]}")
            if current_bullet:  # 實彈
                if self.shortened:
                    self.players[opponent]["health"] -= 2
                    reward = 0.5
                    self.shortened = False
                else:
                    self.players[opponent]["health"] -= 1
                    reward = 0.3
                self.bullets_live -=1
                # print(f"After: Live: {self.bullets_live}, Blank: {self.bullets_blank}.\n Opponent: {opponent},health:{self.players[opponent]}")

            else:  # 空彈
                self.bullets_blank -=1
                reward = -0.1
            
            if self.players[opponent]["skip"] == True:
                self.players[opponent]["skip"] = False
            else:
                if self.players[opponent]["skip"] == True:
                    self.players[opponent]["skip"] = False
                else:
                    self.turn = 1 - self.turn
            self.magnifier_result = None # 開槍後清空提示

            # if self.players[opponent]["skip"] == 0:
            #     self.turn = 1 - self.turn  # 如果對方沒被手銬就換人
            # elif self.players[opponent]["skip"] < 0:
            #     return ValueError("不應該出現負數")
            # else: self.players[opponent]["skip"] -= 1  # 被手銬後下一輪

        elif ACTION_SPACE[action] == "shoot_self":
            current_bullet = self.bulletsList.pop(0)
            if current_bullet:  # 實彈
                if self.shortened:
                    self.players[player]["health"] -= 2
                    reward = -0.5
                    self.shortened = False
                else:
                    self.players[player]["health"] -= 1
                    reward = -0.3
                self.bullets_live -=1
                if self.players[opponent]["skip"] == True:
                    self.players[opponent]["skip"] = False
                else:
                    self.turn = 1 - self.turn 

            else:  # 空彈
                self.bullets_blank -=1
                reward = 0.2
                # 這裡你可以決定：要不要換人？
                # 如果規則是空彈自己可繼續 -> 不換人
                # 如果規則是仍然換人 -> self.turn = 1 - self.turn
            # if self.players[opponent]["skip"] == 0:
            #     self.turn = 1 - self.turn  # 如果對方沒被手銬就換人
            # elif self.players[opponent]["skip"] < 0:
            #     return ValueError("不應該出現負數")
            # else: self.players[opponent]["skip"] -= 1  # 被手銬後不換人
            self.magnifier_result = None # 開槍後清空提示

        # 道具的部分你可以再往下加分支
        # V1_props = ["magnifier","cigarette","beer","saw","handcuffs"]
        elif ACTION_SPACE[action] == "use_item_magnifier": 
            if "magnifier" in self.players[player]["items"]:
                self.players[player]["items"].remove("magnifier")
                self.magnifier_result = self.bulletsList[0]  
            else : reward = -1         
        elif ACTION_SPACE[action] == "use_item_cigarette":
            if "cigarette" in self.players[player]["items"]:
                self.players[player]["items"].remove("cigarette")
                if self.players[player]["health"]< self.maxHealth:
                    self.players[player]["health"] +=1
                else: reward = -0.1
            else : reward = -1         
        elif ACTION_SPACE[action] == "use_item_beer":
            if "beer" in self.players[player]["items"]:
                self.players[player]["items"].remove("beer")
                current_bullet = self.bulletsList.pop(0)
                if current_bullet:
                    self.bullets_live -=1
                else:
                    self.bullets_blank -=1
            else : reward = -1         
        elif ACTION_SPACE[action] == "use_item_saw":
            if "saw" in self.players[player]["items"]:
                self.players[player]["items"].remove("saw")
                self.shortened = True
            else : reward = -1         
        elif ACTION_SPACE[action] == "use_item_handcuffs":
            if "handcuffs" in self.players[player]["items"]:
                self.players[player]["items"].remove("handcuffs")
                self.players[opponent]["skip"] = True
            else : reward = -1         
        else:
            return ValueError(f"Action: {action} not found")
        
        if self.players[player]["health"] <= 0:
            done = True
            reward = -1
            info["reason"] = f"player{player} dead"
        if self.players[opponent]["health"] <= 0:
            done = True
            reward = 1
            info["reason"] = f"player{opponent} dead"
        obs = self.get_state()
        return obs, reward, done, info

    
    def getRandItem(self,amount):
        return random.choices(allowItems,k=amount)
    
    def newRound(self):
        # 抽新道具
        self.players[0]["items"] += self.getRandItem(min(self.itemPerRound, 8 - len(self.players[0]["items"])))
        self.players[1]["items"] += self.getRandItem(min(self.itemPerRound, 8 - len(self.players[1]["items"])))

        # 重抽子彈
        total = random.randint(3, 8)
        # red 至少 1，gray 至少 1
        red = random.randint(1, total - 1)
        gray = total - red
        Bullet = {"live": red, "blank": gray}
        self.bulletsList = random.sample([True]*(Bullet["live"]) + [False]*(Bullet["blank"]), total)
        
        self.bullets_blank = Bullet["blank"]
        self.bullets_live = Bullet["live"]


    def get_action_mask(self):
        # ACTION_SPACE = {
        #     0: "shoot_opponent",
        #     1: "shoot_self",
        #     2: "use_item_magnifier",
        #     3: "use_item_cigarette",
        #     4: "use_item_beer",
        #     5: "use_item_saw",
        #     6: "use_item_handcuffs"  
        # }
        player = self.turn
        mask = [1] * len(ACTION_SPACE)  # 預設全部可選

        # 沒有道具就不能使用對應動作
        if "magnifier" not in self.players[player]["items"] or self.magnifier_result != None:
            mask[2] = 0
        if "cigarette" not in self.players[player]["items"]:
            mask[3] = 0
        if "beer" not in self.players[player]["items"]:
            mask[4] = 0
        if "saw" not in self.players[player]["items"] or self.shortened == True:
            mask[5] = 0
        if "handcuffs" not in self.players[player]["items"] or self.players[1-player]["skip"] == True:
            mask[6] = 0

        return mask


if __name__ == "__main__":
    print("You are running the game script. It is for testing")
    env = GameEnv()
    env.reset()

    done = False

    while(done == False):
        print(f"You are: {env.turn}\nYour state:")
        print(env.players[env.turn])
        print("Oppoment's state:")
        print(env.players[1 - env.turn])
        move = int(input(f"Enter your move. Avalible: {env.get_action_mask()}"))
        obs, reward, done, info = env.step(move)
        print(reward)
