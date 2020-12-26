import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from agents import zero, experience
from agents.encoders import zeroencoder

from game import kerasutil
kerasutil.set_gpu_memory_dynamic()

from tensorflow.keras.layers import Activation, Dense, Input, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

import argparse
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--num-rounds', type=int, default=1600)
    parser.add_argument('output_file')
    args = parser.parse_args()

    encoder = zeroencoder.ZeroEncoder(args.board_size)

    board_input = Input(shape=encoder.shape(), name='board_input')
    pb = board_input
    for _ in range(4):
        pb = Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu')(pb)
    
    policy_conv = Conv2D(2, (1, 1), data_format='channels_first', activation='relu')(pb)
    policy_flat = Flatten()(policy_conv)
    policy_output = Dense(encoder.num_actions(), activation='softmax')(policy_flat)

    value_conv = Conv2D(1, (1, 1), data_format='channels_first', activation='relu')(pb)
    value_flat = Flatten()(value_conv)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    model = Model(inputs=[board_input], outputs=[policy_output, value_output])
    opt = SGD(lr=0.02)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    new_agent = zero.ZeroAgent(model, encoder, args.num_rounds, args.temperature)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.save(outf)

if __name__ == '__main__':
    main()