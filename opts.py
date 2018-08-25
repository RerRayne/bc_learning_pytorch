import os
import argparse


def parse():
    def get_degault_milistones(nEpochs):
        return [int(0.5 * nEpochs), int(0.75 * nEpochs)]

    parser = argparse.ArgumentParser(description='BC learning for sounds')

    # General settings
    parser.add_argument('--dataset', required=True, choices=['esc10', 'esc50'])
    parser.add_argument('--netType', required=True, choices=['EnvNet'])
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--split', type=int, default=-1, help='esc: 1-5, urbansound: 1-10 (-1: run on all splits)')
    parser.add_argument('--save', default='None', help='Directory to save the results')
    parser.add_argument('--sr', type=int, default=16000, help="Audio sample rate")
    parser.add_argument('--gpu', type=int, default=-1)

    # Learning settings (default settings are defined below)
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--nEpochs', type=int, default=-1)
    parser.add_argument('--LR', type=float, default=-1, help='Initial learning rate')
    parser.add_argument('--milestones', type=float, nargs='*', default=-1, help='When decrease LR')
    parser.add_argument('--gamma', type=float, default=0.1, help='decreasing coeff')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--weightDecay', type=float, default=5e-4)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)

    opt = parser.parse_args()

    # Dataset details
    if opt.dataset == 'esc50':
        opt.nClasses = 50
        opt.nFolds = 5
    elif opt.dataset == 'esc10':
        opt.nClasses = 10
        opt.nFolds = 5

    if opt.split == -1:
        opt.splits = range(1, opt.nFolds + 1)
    else:
        opt.splits = [opt.split]

    # Model details
    if opt.netType == 'EnvNet':
        opt.fs = 16000
        opt.inputLength = 24014

    # Default settings (nEpochs will be doubled if opt.BC)
    default_settings = dict()
    milistones = get_degault_milistones(opt.nEpochs)
    default_settings['esc50'] = {'EnvNet': {'nEpochs': 600, 'LR': 0.01, 'milestones': milistones}}
    default_settings['esc10'] = {'EnvNet': {'nEpochs': 600, 'LR': 0.01, 'milestones': milistones}}

    for key in ['nEpochs', 'LR', 'milestones']:
        if eval('opt.{}'.format(key)) == -1:
            setattr(opt, key, default_settings[opt.dataset][opt.netType][key])
            if key == 'nEpochs' and opt.BC:
                opt.nEpochs *= 2
                opt.milestones = get_degault_milistones(opt.nEpochs)

    if opt.save != 'None' and not os.path.isdir(opt.save):
        os.makedirs(opt.save)

    display_info(opt)

    return opt


def display_info(opt):
    if opt.BC:
        learning = 'BC'
    else:
        learning = 'standard'

    print('+------------------------------+')
    print('| Sound classification')
    print('+------------------------------+')
    print('| gpu: {}'.format(opt.gpu))
    print('| dataset  : {}'.format(opt.dataset))
    print('| sample rate  : {}'.format(opt.sr))
    print('| netType  : {}'.format(opt.netType))
    print('| learning : {}'.format(learning))
    print('| nEpochs  : {}'.format(opt.nEpochs))
    print('| LRInit   : {}'.format(opt.LR))
    print('| batchSize: {}'.format(opt.batchSize))
    print('| nesterov   : {}'.format(opt.nesterov))
    print('| milestones : {}'.format(opt.milestones))
    print('| gamma : {}'.format(opt.gamma))
    print('+------------------------------+')
