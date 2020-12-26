import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse

import h5py

from agents import zero
from agents import experience as exp
from game import kerasutil
kerasutil.set_gpu_memory_dynamic()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', required=True)
    parser.add_argument('--agent-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()

    learning_agent = zero.ZeroAgent.load(h5py.File(args.learning_agent, 'r'))
    for exp_filename in args.experience:
        print('Training with %s...' % exp_filename)
        exp_buffer = exp.ExperienceBuffer.load(h5py.File(exp_filename, 'r'))
        learning_agent.train(exp_buffer, lr=args.lr, batch_size=args.bs, clipnorm=1)

    with h5py.File(args.agent_out, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    main()