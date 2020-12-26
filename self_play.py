import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from game.kerasutil import set_gpu_memory_dynamic
set_gpu_memory_dynamic()

import multiprocessing
import random
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np
import argparse

from agents import zero
from agents import experience as exp
from game import kerasutil
from game.omok.type import Player, Point
from game.omok.board import State


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    Player.black: ' x ',
    Player.white: ' o ',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.num_cols]))


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player, board_size):
    moves = []
    game = State.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_action(game)
        moves.append(next_move)
        game = game.apply(next_move)

    print_board(game.board)
    game_result = game.result()
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result,
        margin=0,
    )


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train', suffix='.hdf5')
    os.close(fd)
    return fname


def do_self_play(board_size, agent_filename,
                 num_games, temperature,
                 experience_filename,
                 gpu_frac):
    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = zero.ZeroAgent.load(h5py.File(agent_filename, 'r'))
    agent2 = zero.ZeroAgent.load(h5py.File(agent_filename, 'r'))

    agent1.set_temperature(temperature)
    agent2.set_temperature(temperature)

    collector1 = exp.ExperienceCollector()
    collector2 = exp.ExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins.')
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        elif game_record.winner == color1.next:
            print('Agent 2 wins.')
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)
        else:
            print('Draw.')
            collector2.complete_episode(reward=0)
            collector1.complete_episode(reward=0)
        color1 = color1.next

    experience = exp.combine_experience([collector1, collector2])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.save(experience_outf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--experience-out', '-o', required=True)
    parser.add_argument('--num-workers', '-w', type=int, default=1)
    parser.add_argument('--temperature', '-t', type=float, default=2.0)
    parser.add_argument('--board-size', '-b', type=int, default=19)

    args = parser.parse_args()

    experience_files = []
    workers = []
    gpu_frac = 0.95 / float(args.num_workers)
    games_per_worker = args.num_games // args.num_workers
    print('Starting workers...')
    for i in range(args.num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                args.board_size,
                args.learning_agent,
                games_per_worker,
                args.temperature,
                filename,
                gpu_frac,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()

    # Merge experience buffers.
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    combined_buffer = exp.ExperienceBuffer.load(h5py.File(first_filename, 'r'))
    for filename in other_filenames:
        next_buffer = exp.ExperienceBuffer.load(h5py.File(filename, 'r'))
        combined_buffer = exp.combine_experience([combined_buffer, next_buffer])
    print('Saving into %s...' % args.experience_out)
    with h5py.File(args.experience_out, 'w') as experience_outf:
        combined_buffer.save(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


if __name__ == '__main__':
    main()