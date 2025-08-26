import random
import numpy as np

SUITS = ['C', 'D', 'H', 'S']  # 梅花、方塊、紅心、黑桃
RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']

# 編碼：card_int = rank_idx * 4 + suit_idx
def card_to_int(rank_idx, suit_idx):
    return rank_idx * 4 + suit_idx

def int_to_card(card_int):
    rank_idx = card_int // 4
    suit_idx = card_int % 4
    return RANKS[rank_idx], SUITS[suit_idx]

def card_value(card_int):
    rank_idx = card_int // 4
    suit_idx = card_int % 4
    return (rank_idx, suit_idx)

def compare_cards(card1, card2):
    # 回傳 True 如果 card1 比 card2 大
    v1 = card_value(card1)
    v2 = card_value(card2)
    if v1[0] != v2[0]:
        return v1[0] > v2[0]
    return v1[1] > v2[1]

class BigTwoEnv:
    def __init__(self, num_players=4):
        assert 2 <= num_players <= 4
        self.num_players = num_players
        self.deck = [i for i in range(52)]
        self.hands = [[] for _ in range(num_players)]
        self.current_player = None
        self.current_combo = []
        self.pass_count = 0
        self.history = []
        self.done = False

    def reset(self):
        self.done = False
        self.pass_count = 0
        self.current_combo = []
        self.history = []
        self.deck = [i for i in range(52)]
        random.shuffle(self.deck)

        # 發牌
        cards_per_player = len(self.deck) // self.num_players
        self.hands = [self.deck[i*cards_per_player:(i+1)*cards_per_player] for i in range(self.num_players)]

        # 找到梅花3（♣3）作為起始牌
        club_three = card_to_int(0, 0)
        for i, hand in enumerate(self.hands):
            if club_three in hand:
                self.current_player = i
                break

        # 排序手牌
        for hand in self.hands:
            hand.sort(key=lambda c: (c // 4, c % 4))

        return self.get_obs()

    def get_obs(self):
        obs = {
            'hand': self.cards_to_onehot(self.hands[self.current_player]),
            'current_combo': self.cards_to_onehot(self.current_combo),
            'pass_count': self.pass_count,
            'player_id': self.current_player,
            'hand_counts': [len(h) for h in self.hands],
        }
        return obs

    def cards_to_onehot(self, cards):
        vec = np.zeros(52, dtype=np.int8)
        for c in cards:
            vec[c] = 1
        return vec

    def compute_valid_actions(self):
        hand = self.hands[self.current_player]
        valid = [0]  # 0 = pass
        for card in hand:
            if self.can_beat([card], self.current_combo):
                valid.append(card + 1)  # +1 offset，避免與 pass 衝突
        return valid

    def can_beat(self, new_combo, old_combo):
        # 僅支援單張出牌
        if not old_combo:
            if self.history == []:
                # 首出必須包含梅花3
                club_three = card_to_int(0, 0)
                return club_three in new_combo
            return True
        if len(new_combo) != len(old_combo):
            return False
        if len(new_combo) == 1:
            return compare_cards(new_combo[0], old_combo[0])
        return False  # TODO: 加入雙張/五張等

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Please reset.")

        if action == 0:
            self.pass_count += 1
            self.history.append((self.current_player, []))

            # 🛠️ 修正：若其他人都 pass，清空牌組
            if self.pass_count >= self.num_players - 1 and self.current_combo:
                self.current_combo = []
                self.pass_count = 0

        else:
            card = action - 1
            if card not in self.hands[self.current_player]:
                raise ValueError(f"Card {card} not in hand.")
            if not self.can_beat([card], self.current_combo):
                raise ValueError(f"Card {card} cannot beat current combo.")

            self.hands[self.current_player].remove(card)
            self.current_combo = [card]
            self.pass_count = 0
            self.history.append((self.current_player, [card]))

            if len(self.hands[self.current_player]) == 0:
                if sum(len(h) > 0 for h in self.hands) == 1:
                    self.done = True

        self.current_player = (self.current_player + 1) % self.num_players
        return self.get_obs(), 0, self.done, {}


    def render(self):
        print(f"\n🎮 玩家 {self.current_player} 的回合")
        print("當前場上牌：", [int_to_card(c) for c in self.current_combo])
        for i, hand in enumerate(self.hands):
            print(f"玩家 {i} 手牌數：{len(hand)}")

# 測試用主程式
def test_env():
    env = BigTwoEnv(num_players=4)
    obs = env.reset()

    print("=== 遊戲開始 ===")
    for i, hand in enumerate(env.hands):
        cards_str = [f"{r}{s}" for (r, s) in map(int_to_card, hand)]
        print(f"玩家 {i} 手牌: {cards_str}")
    print(f"起始牌玩家：玩家 {env.current_player}")

    done = False
    step_count = 0

    while not done and step_count < 100:
        valid_actions = env.compute_valid_actions()
        print(f"\n玩家 {env.current_player} 可選動作: {valid_actions}")

        # 策略：選第一張合法牌，不然 pass
        if len(valid_actions) > 1:
            action = valid_actions[1]
        else:
            action = 0

        print(f"玩家 {env.current_player} 選擇動作: {action}")

        try:
            obs, reward, done, info = env.step(action)
        except ValueError as e:
            print("⚠️ 動作錯誤:", e)
            break

        print(f"當前牌組: {[int_to_card(c) for c in env.current_combo]}")
        for i, hand in enumerate(env.hands):
            print(f"玩家 {i} 手牌數: {len(hand)}")

        step_count += 1

    print("\n=== 遊戲結束 ===")
    for i, hand in enumerate(env.hands):
        print(f"玩家 {i} 最終手牌數：{len(hand)}")

# 執行測試
if __name__ == "__main__":
    test_env()
