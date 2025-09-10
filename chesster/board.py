# 簡化版 SAN 解析與 ASCII 棋盤
# 僅支援兵、馬、象、車、后、王移動 + 吃子 + 易位
# 不支援升變/模糊解法（多子同格可移動時需更完整處理）

import re

FILES = "abcdefgh"
RANKS = "12345678"

def init_board():
    return [
        list("rnbqkbnr"),
        list("pppppppp"),
        list("........"),
        list("........"),
        list("........"),
        list("........"),
        list("PPPPPPPP"),
        list("RNBQKBNR"),
    ]

def print_board(board):
    for r in range(8):
        row = board[r]
        print(8-r, " ".join(row))
    print("  a b c d e f g h\n")

board = init_board()
print_board(board)

done = False

while done == False:
    moves = input("Enter single PGN move: ")
    print(moves)
    done = True
    