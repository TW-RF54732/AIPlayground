import BigTwoEnv

def test_env():
    env = BigTwoEnv(num_players=4)
    obs = env.reset()

    print("=== 遊戲開始 ===")
    for i, hand in enumerate(env.hands):
        cards_str = [f"{r}{s}" for (r, s) in map(int_to_card, hand)]
        print(f"玩家 {i} 手牌: {cards_str}")
    print(f"起始牌玩家: 玩家 {env.current_player}")

    done = False
    step_count = 0

    while not done and step_count < 100:
        valid_actions = env.compute_valid_actions()
        print(f"\n玩家 {env.current_player} 可行動: {valid_actions}")

        # 簡單策略: 選第一個合法非pass牌，若沒有出牌則pass
        if len(valid_actions) > 1:
            action = valid_actions[1]
        else:
            action = 0

        print(f"玩家 {env.current_player} 選擇動作: {action}")

        try:
            obs, reward, done, info = env.step(action)
        except ValueError as e:
            print("動作錯誤:", e)
            break

        print(f"當前牌組: {[int_to_card(c) for c in env.current_combo]}")
        for i, hand in enumerate(env.hands):
            print(f"玩家 {i} 手牌數: {len(hand)}")

        step_count += 1

    print("=== 遊戲結束 ===")
    for i, hand in enumerate(env.hands):
        print(f"玩家 {i} 最終手牌數: {len(hand)}")

test_env()
