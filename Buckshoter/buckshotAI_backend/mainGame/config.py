import os,json
itemJsonPath = os.path.join(os.path.dirname(__file__),"setItem.json")
def setItem(state):
    with open(itemJsonPath, "r", encoding="utf-8") as f:
        data = json.load(f)  # 解析 JSON
        p1Item = data["player1"]
        p2Item = data["player1"]
        return p1Item,p2Item#p1,p2
        

def setHealth(state):
    health = 4
    maxHP = 4
    return health,health,maxHP #p1,p2

