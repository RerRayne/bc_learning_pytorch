import os

import numpy as np
import time
import torch
import pickle as pkl

from torch.nn.modules.loss import KLDivLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data

import models
import opts
from datasets import BCDatasets, tensor_to_numpy, get_train_transform, get_test_transform, IMG, LABEL
from draw_process import draw_progress
from models import weights_init

DoubleTensor = None
FloatTensor = None
LongTensor = None
ByteTensor = None


class iter_len:
    def __init__(self, enumerator, l):
        self.enumerator = enumerator
        self.l = l

    def __next__(self):
        return self.enumerator.__next__()

    def __len__(self):
        return self.l

    def __iter__(self):
        return self.enumerator


def cudify_i(x):
    try:
        iter(x)
    except:
        return

    for a in x:
        yield cudify(a)


def cudify(x):
    if isinstance(x, tuple):
        return (cudify(a) for a in x)
    if isinstance(x, list):
        return [cudify(a) for a in x]
    if isinstance(x, dict):
        return dict([(k, cudify(x[k])) for k in x])
    if isinstance(x, torch.Tensor):
        return x.cuda() if torch.cuda.is_available() else x
    if '__len__' in dir(x):
        return iter_len(cudify_i(x), x.__len__())

    as_i = cudify_i(x)
    if as_i == False:
        return x.cuda() if torch.cuda.is_available() else x
    return as_i


def accuracy(prediction, target):
    with torch.no_grad():
        _, pred = prediction.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = torch.mean(correct.view(-1).float())
        return result


def use_cuda(opt):
    return opt.gpu > -1 and torch.cuda.is_available()


def check_cuda(opt):
    global DoubleTensor, FloatTensor, LongTensor, ByteTensor

    if use_cuda(opt):
        DoubleTensor = torch.cuda.DoubleTensor
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        ByteTensor = torch.cuda.ByteTensor
    else:
        DoubleTensor = torch.DoubleTensor
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor
        ByteTensor = torch.ByteTensor


def create_model(opt):
    model = getattr(models, opt.netType)(opt.nClasses)
    if use_cuda(opt):
        print("we will use cuda model!")
        model = model.cuda()
        model.apply(weights_init)
    return model


def create_optimizer(model, opt):
    opt_params = {"weight_decay": opt.weightDecay,
                  "momentum": opt.momentum,
                  "nesterov": opt.nesterov,
                  "lr": opt.LR}
    return SGD(model.parameters(), **opt_params)


def get_data_generators(opt, test_fold):
    folds = set(range(1, opt.nFolds + 1))
    exclude_train = {test_fold}
    exclude_val = folds - exclude_train

    train_set = BCDatasets(opt.data,
                           opt.dataset,
                           opt.sr,
                           exclude_train,
                           transform=get_train_transform(opt.inputLength),
                           mix=opt.BC)
    val_set = BCDatasets(opt.data,
                         opt.dataset,
                         opt.sr,
                         exclude_val,
                         transform=get_test_transform(opt.inputLength))

    params = {'batch_size': opt.batchSize,
              'shuffle': True,
              'num_workers': 1}
    training_generator = data.DataLoader(train_set, **params)

    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 1}
    validation_generator = cudify(data.DataLoader(val_set, **params))

    if use_cuda(opt):
        training_generator = cudify(training_generator)
        validation_generator = cudify(validation_generator)

    return training_generator, validation_generator


def train(model, optimizer, loss, training_generator):
    # train model
    model.train(True)
    epoch_train_loss = []
    epoch_error_rate = 0.0

    for batch in training_generator:
        features, labels = batch[IMG], batch[LABEL]
        print(features.shape)
        output = model.forward(features)

        loss = loss(output, labels.type(FloatTensor))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_train_loss.append(tensor_to_numpy(loss.data).reshape((1))[0])

        labels = labels.argmax(dim=1).reshape((-1))
        error = tensor_to_numpy(accuracy(output, labels)) * labels.size()[-1]
        epoch_error_rate += error

    model.train(False)

    return model, optimizer, epoch_train_loss, epoch_error_rate


def validate(model, validation_generator):
    epoch_val_loss = []
    val_epoch_rate = 0.0
    for batch in validation_generator:
        features, labels = batch[IMG].reshape((-1, 1, 1, opt.inputLength)), batch[LABEL]
        output = model.forward(features).mean(dim=0).reshape((1, 10))

        loss = kl_loss(output, labels.type(FloatTensor))

        epoch_val_loss.append(tensor_to_numpy(loss.data).reshape((1))[0])

        labels = labels.argmax().reshape((-1))
        val_epoch_rate += tensor_to_numpy(accuracy(output, labels)) * labels.size()[-1]

    return epoch_val_loss, val_epoch_rate


def accuracy_on_batch(error_rate, generator, batch_size):
    return 100.0 * (1.0 - error_rate / (len(generator) * batch_size))


if __name__ == '__main__':
    opt = opts.parse()

    check_cuda(opt)

    global_train_loss = []
    global_val_loss = []
    global_train_error = []
    global_val_error = []
    n_folds = opt.nFolds

    for test_fold in opt.splits:
        model = create_model(opt)
        optimizer = create_optimizer(model, opt)
        scheduler = MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.1)

        kl_loss = KLDivLoss()

        train_loss = []
        train_error_rate = []
        val_loss = []
        val_error_rate = []

        for epoch in range(opt.nEpochs):
            scheduler.step()
            training_generator, validation_generator = get_data_generators(opt, test_fold)

            start_time = time.time()

            model, optimizer, epoch_train_loss, epoch_error_rate = train(model,
                                                                         optimizer,
                                                                         kl_loss,
                                                                         training_generator)

            epoch_val_loss, val_epoch_rate = validate(model, validation_generator)

            train_loss.append(np.mean(epoch_train_loss))
            val_loss.append(np.mean(epoch_val_loss))

            train_error_rate.append(accuracy_on_batch(epoch_error_rate, training_generator, opt.batchSize))
            val_error_rate.append(accuracy_on_batch(val_epoch_rate, validation_generator, 1))

        torch.save(model, os.path.join(opt.save, "model_{}.bin".format(test_fold)))
        draw_progress(train_loss, val_loss,
                      train_error_rate, val_error_rate,
                      epoch,
                      test_fold, opt.save)
        global_train_loss.append(train_loss)
        global_train_error.append(train_error_rate)

        global_val_loss.append(val_loss)
        global_val_error.append(val_error_rate)

    report = {
                "train_loss": global_train_loss,
                "train_acc": global_train_error,
                "val_loss": global_val_loss,
                "val_acc": global_val_error,
                "opt": opt
    }

    with open(os.path.join(opt.save, "report.pkl"), "wb") as f:
        pkl.dump(report, f)
