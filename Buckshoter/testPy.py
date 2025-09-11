import game

env = game.GameEnv()

env.reset()
env.startUp()

state = env.get_state()
fState = state["players"]["player1"]["items"]
sfState = state["players"]["player2"]["items"]

# print(fState,end="\n")
# print(sfState)

print(state)