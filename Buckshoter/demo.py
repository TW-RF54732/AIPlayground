import random


class BuckshotRoulette:
    def __init__(self, hp=3):
        # 公共數據
        self.round = 1
        # 子彈隨機裝填：live=實彈, blank=空彈
        self.bullets = random.sample(["live", "blank"] * 3, 6)
        self.current_player = 0  # 0=玩家A, 1=玩家B
        self.is_over = False

        # 玩家數據
        self.players = [
            {"hp": hp, "items": ["cigarette", "magnifier"]},
            {"hp": hp, "items": ["saw", "beer"]}
        ]

    def use_item(self, player_id, item):
        if item not in self.players[player_id]["items"]:
            print(f"玩家{player_id} 想用 {item}，但沒有這個道具！")
            return
        print(f"玩家{player_id} 使用了 {item}")
        self.players[player_id]["items"].remove(item)

        # 簡單道具效果（僅示範，不是真正遊戲規則）
        if item == "cigarette":
            self.players[player_id]["hp"] = min(3, self.players[player_id]["hp"] + 1)
        elif item == "saw":
            if self.bullets:
                self.bullets[-1] = "live"  # 保證最後一顆是實彈
        elif item == "magnifier":
            if self.bullets:
                print(f"🔍 玩家{player_id} 偷看了下一顆子彈: {self.bullets[0]}")
        elif item == "beer":
            if self.bullets:
                thrown = self.bullets.pop(0)
                print(f"🍺 玩家{player_id} 丟掉了 {thrown} 彈")

    def shoot(self, player_id, target_id):
        if not self.bullets:
            print("沒有子彈了！遊戲平局")
            self.is_over = True
            return

        bullet = self.bullets.pop(0)
        print(f"玩家{player_id} 對 玩家{target_id} 開槍 -> {bullet}")

        if bullet == "live":
            self.players[target_id]["hp"] -= 1
            print(f"💥 玩家{target_id} 受傷！剩餘 HP={self.players[target_id]['hp']}")

        if self.players[target_id]["hp"] <= 0:
            self.is_over = True
            print(f"☠️ 玩家{target_id} 死亡，玩家{player_id} 勝利！")
            return

        # 換回合
        self.current_player = 1 - player_id
        self.round += 1

    def play_turn(self):
        pid = self.current_player
        opponent = 1 - pid

        print(f"\n--- 回合 {self.round} (玩家{pid}) ---")
        print(f"玩家狀態: A(HP={self.players[0]['hp']}, 道具={self.players[0]['items']}) | "
              f"B(HP={self.players[1]['hp']}, 道具={self.players[1]['items']})")
        print(f"剩餘子彈: {len(self.bullets)}")

        # 隨機決定動作
        actions = ["shoot", "item"] if self.players[pid]["items"] else ["shoot"]
        action = random.choice(actions)

        if action == "item":
            item = random.choice(self.players[pid]["items"])
            self.use_item(pid, item)

            # 使用完道具後 **隨機決定要不要再開槍**
            if random.random() < 0.7:  # 70% 機率開槍
                self.shoot(pid, opponent)
        else:
            self.shoot(pid, opponent)


if __name__ == "__main__":
    game = BuckshotRoulette()

    while not game.is_over:
        game.play_turn()
