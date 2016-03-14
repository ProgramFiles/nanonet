#!/usr/bin/env python
import argparse
import json
import sys

import numpy as np
from nanonet import nn
from nanonet.cmdargs import FileExist

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Convert currennt json network file into pickle',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('input', action=FileExist,
    help='File containing current network')
parser.add_argument('output', help='Output pickle file')

def toarray(x):
    return np.ascontiguousarray(np.array(x, order='C', dtype=nn.tang_nn_type))

def parse_layer_input(size, weights):
    return None

def parse_layer_feedforward(size, weights, fun):
    M = toarray(weights['input'])
    M = M.reshape((size, -1)).transpose()
    b = toarray(weights['bias'])
    return nn.layer(M, b, fun)

def parse_layer_feedforward_tanh(size, weights):
    return parse_layer_feedforward(size, weights, nn.tanh)

def parse_layer_feedforward_sigmoid(size, weights):
    return parse_layer_feedforward(size, weights, nn.sigmoid)

def parse_layer_feedforward_linear(size, weights):
    return parse_layer_feedforward(size, weights, nn.linear)

def parse_layer_softmax(size, weights):
    M = toarray(weights['input'])
    M = M.reshape((size, -1)).transpose()
    b = toarray(weights['bias'])
    return nn.softmax(M ,b)

def parse_layer_multiclass(size, weights):
    return None

def parse_layer_blstm(size, weights):
    size = size / 2
    wgts_input = toarray(weights['input']).reshape((4, 2, size, -1)).transpose((0, 1, 3, 2))
    wgts_bias = toarray(weights['bias']).reshape((4, 2, -1))
    wgts_internalMat = toarray(weights['internal'][: 4 * size * size * 2]).reshape((4, 2, size, size)).transpose((0, 1, 3, 2))
    wgts_internalPeep = toarray(weights['internal'][4 * size * size * 2 :]).reshape((3, 2, size))

    iM1 = wgts_input[:, 0, :, :]
    bM1 = wgts_bias[:, 0, :]
    lM1 = wgts_internalMat[:, 0, :, :]
    pM1 = wgts_internalPeep[:, 0, :]
    layer1 = nn.lstm_layer(iM1, lM1, bM1, pM1)

    iM2 = wgts_input[:, 1, :, :]
    bM2 = wgts_bias[:, 1, :]
    lM2 = wgts_internalMat[:, 1, :, :]
    pM2 = wgts_internalPeep[:, 1, :]
    layer2 = nn.lstm_layer(iM2, lM2, bM2, pM2)
    return nn.birnn(layer1, layer2)

def parse_layer_lstm(size, weights):
    iM = toarray(weights['input']).reshape((4, size, -1)).transpose((0, 2, 1))
    bM = toarray(weights['bias']).reshape((4, size))
    lM = toarray(weights['internal'][ : 4 * size * size]).reshape((4, size, size)).transpose((0, 2, 1))
    pM = toarray(weights['internal'][4 * size * size : ]).reshape((3, size))
    return nn.lstm(iM, lM, bM, pM)


LAYER_DICT = {'input' : parse_layer_input,
              'blstm' : parse_layer_blstm,
              'feedforward_tanh' : parse_layer_feedforward_tanh,
              'feedforward_logistic' : parse_layer_feedforward_sigmoid,
              'feedforward_identity' : parse_layer_feedforward_linear,
              'lstm' : parse_layer_lstm,
              'blstm' : parse_layer_blstm,
              'softmax' : parse_layer_softmax,
              'multiclass_classification' : parse_layer_multiclass}

def parse_layer(layer_type, size, weights):
    if not layer_type in LAYER_DICT:
        sys.stderr.write('Unsupported layer type {}.\n'.format(layer_type))
        exit(1)
    return LAYER_DICT[layer_type](size, weights)

if __name__ == '__main__':
    args = parser.parse_args()

    try:
        with open(args.input, 'r') as fh:
            in_network = json.load(fh)
    except:
        sys.stderr.write('Failed to read from {}.\n')
        exit(1)

    if not 'layers' in in_network:
        sys.stderr.write('Could not find any layers in {} -- is it a network file?\n'.format(args.network))
        exit(1)
    if not 'weights' in in_network:
        sys.stderr.write('Could not find any weights in {} -- is network trained?\n'.format(args.network))
        exit(1)

    layers = list()
    for layer in in_network['layers']:
        wgts = in_network['weights'][layer['name']] if layer['name'] in in_network['weights'] else None
        layers.append(parse_layer(layer['type'], layer['size'], wgts))
    layers = filter(lambda x: x is not None, layers)

    network = nn.serial(layers)
    np.save(args.output, network)
