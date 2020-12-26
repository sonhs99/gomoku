import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from agents import naive
from game.omok import board
from game.omok import type
from game.omok.utils import print_board, print_move
import time

from game.kerasutil import set_gpu_memory_dynamic
from tensorflow.keras.models import load_model, save_model

import h5py

set_gpu_memory_dynamic()

def main():
    board_size = 15
    game = board.State.new_game(board_size)
    bots = {
        type.Player.black : naive.RandomBot(),
        type.Player.white : naive.RandomBot(),
    }
    while not game.is_over():
        time.sleep(0.3)

        print(chr(27) + '[2j')
        print_board(game.board)
        bot_move = bots[game.next_player].select_action(game)
        print_move(game.next_player, bot_move)
        game = game.apply(bot_move)

    print_board(game.board)
    result = game.result()
    print(result, 'win')

if __name__ == '__main__':
    main()