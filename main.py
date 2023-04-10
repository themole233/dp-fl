# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import matplotlib
import matplotlib.pyplot as plt
import copy
import random
import numpy as np
from torchvision import datasets, transforms
import torch
import math
from torch import nn, autograd
from scipy.special import comb, perm
import math
from functools import reduce
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar, SampleConvNet
from models.Fed import FedAvg, FedAvgCollect
from models.test import test_img
from torch.utils.data import DataLoader, Dataset
import os

from utils.model_cnn import vgg11
from copy import deepcopy
import models.Update
import time

matplotlib.use('Agg')

fed_avg_collect = FedAvgCollect()


args = args_parser()
args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.device_c = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

ceshi = [int(10),int(20),int(30),int(40),int(50),int(60),int(70),int(80),int(90)]

def clamp(x, y, z):
    floor = torch.min(x, z).to(args.device_c)
    ceil = torch.max(x, z).to(args.device_c)
    return torch.min(torch.max(floor, y), ceil).to(args.device_c)


torch.cuda.manual_seed(1)


acc_test_repro = []
acc_train_repro = []


if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST(root="./datasets", train=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(root="./datasets", train=False, transform=trans_mnist)


    print('iid:', args.iid)
    if args.iid:
        dataset_dict_clients = mnist_iid(dataset_train, args.num_users)
    else:
        dataset_dict_clients = mnist_noniid(dataset_train, args.num_users)
elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    if args.iid:
        dataset_dict_clients = cifar_iid(dataset_train, args.num_users)
    else:
        exit('Error: only consider IID setting in CIFAR10')
else:
    exit('Error: unrecognized dataset')
img_size = dataset_train[0][0].shape


if args.model == 'cnn' and args.dataset == 'cifar':
    net_glob = CNNCifar(args=args).to(args.device)
elif args.model == 'vgg' and args.dataset == 'cifar':
    net_glob = vgg11(pretrained=False, progress=False).to(args.device)
elif args.model == 'cnn' and args.dataset == 'mnist':
    net_glob = CNNMnist(args=args).to(args.device)
elif args.model == 'dppca' and args.dataset == 'mnist':
    net_glob = SampleConvNet().to(args.device)
elif args.model == 'mlp':
    len_in = 1
    for x in img_size:
        len_in *= x
    net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
else:
    exit('Error: unrecognized model')

net_glob.train()

w_global = net_glob.state_dict()
g_global = dict()

key = list(w_global.keys())

for name, parms in net_glob.named_parameters():
    g_global[name] = torch.zeros_like(parms.data)

Expectation_g_2 = dict()
for name, parms in net_glob.named_parameters():
    Expectation_g_2[name] = torch.zeros_like(parms.data).to(args.device_c)




for k, v in g_global.items():
    Expectation_g_2[k] = (0.1 * v * v).to(args.device_c)

L = torch.zeros(args.num_users)
delta_locals = torch.zeros(args.num_users)


loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []

alpha = torch.arange(2, 65)
epsilon = args.epsilon

C = args.lcf
beta = args.beta
sigma_0 = args.nl
G = 1e-6

T = 1



delta_0 = []
delta_local_i = []

number_of_all_data = 0
for idx in range(args.num_users):
    number_of_all_data += len(dataset_dict_clients[idx])

number_clients_epoch = max(int(args.frac * args.num_users), 1)

temp_file_list = os.listdir("temp")
for temp_file in temp_file_list:
    os.remove(os.path.join("temp", temp_file))

for k_iter in range(1, args.rounds+1):
    loss_locals = []
    Z_k_idxs_users = np.random.choice(range(args.num_users), number_clients_epoch, replace=False)

    n_Z_k_number_data_epoch = 0
    for idx in Z_k_idxs_users:
        n_Z_k_number_data_epoch += len(dataset_dict_clients[idx])

    client_FL_weights = dict()
    for idx in Z_k_idxs_users:
        client_FL_weights[idx] = len(dataset_dict_clients[idx]) / n_Z_k_number_data_epoch

    w_local_0 = copy.deepcopy(net_glob.state_dict())


    for idx in Z_k_idxs_users:
        net_glob.load_state_dict(w_local_0)


        aa = 0

        for k in range(100):
            net_glob.train()
            lot_size = args.ls
            batch_size = 100
            dataset_local = DatasetSplit(dataset_train, dataset_dict_clients[idx])
            ldr_train = DataLoader(dataset_local, batch_size=batch_size, shuffle=True)
            loss_func = nn.CrossEntropyLoss()
            g = dict()

            local_ep = int(args.local_ep)

            for batch_idx, (images, labels) in enumerate(ldr_train):
                aa += 1

                images, labels = images.to(args.device_c), labels.to(args.device_c)
                if aa == 1:
                    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr)
                    optimizer.zero_grad()
                    log_probs = net_glob(images)
                    w_local = net_glob.state_dict()
                    loss = loss_func(log_probs, labels)
                    loss.backward()
                    g = dict()
                    for name, parms in net_glob.named_parameters():
                        g[name] = parms.grad.to(args.device_c)
                    Expectation_g_2 = dict()
                    iter_param = dict()
                    iter_param_s = dict()
                    iter_param_delta = dict()
                    iter_m = dict()
                    iter_v = dict()
                    for k, v in w_local.items():
                        v = v - args.lr * g[k]
                        w_local[k] = v
                        if args.optimizer == "RMSPROP":
                            Expectation_g_2[k] = 0.1 * g[k] * g[k]
                        elif args.optimizer == "Momentum":
                            iter_param[k] = args.lr * g[k]
                        elif args.optimizer == "Adadelta":
                            Expectation_g_2[k] = 0.1 * g[k] * g[k]
                            iter_param_delta[k] = torch.sqrt(1e-6 / (Expectation_g_2[k] + 1e-6)) * g[k]
                            iter_param_s[k] = 0.1 * iter_param_delta[k] * iter_param_delta[k]
                        elif args.optimizer == "Adam":
                            iter_m[k] = 0.1*g[k]
                            iter_v[k] = 0.001*g[k]*g[k]
                    g_used = g.copy()
                    w_local_previous = w_local.copy()
                    if args.optimizer == "RMSPROP":
                        Expectation_g_2_used = Expectation_g_2.copy()
                    elif args.optimizer == "Momentum":
                        iter_param_used = iter_param.copy()
                    elif args.optimizer == "Adadelta":
                        Expectation_g_2_used = Expectation_g_2.copy()
                        iter_param_s_used = iter_param_s.copy()
                    elif args.optimizer == "Adam":
                        iter_m_used = iter_m.copy()
                        iter_v_used = iter_v.copy()
                    net_glob.load_state_dict(w_local)
                    if aa == int(args.local_ep):
                        break


                else:
                    if args.optimizer == "Momentum":
                        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=0.9)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()
                        loss = loss_func(log_probs, labels)
                        loss.backward()
                        g = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)
                        for k, v in g.items():
                            iter_param[k] = 0.9 * iter_param[k] + args.lr * v
                        for k, v in w_local.items():
                            v = v - iter_param[k]
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            iter_param_used = iter_param.copy()
                            w_local_previous = w_local.copy()
                        net_glob.load_state_dict(w_local)

                        if aa == int(args.local_ep):
                            break

                    elif args.optimizer == "SGD":
                        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()
                        loss = loss_func(log_probs, labels)
                        loss.backward()
                        g = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)
                        for k, v in w_local.items():
                            v = v - args.lr*g[k]
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            w_local_previous = w_local.copy()
                        net_glob.load_state_dict(w_local)
                        if aa == int(args.local_ep):
                            break

                    elif args.optimizer == "Adadelta":
                        optimizer = torch.optim.Adadelta(net_glob.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()
                        loss = loss_func(log_probs, labels)
                        loss.backward()
                        g = dict()
                        iter_param_delta = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)
                        for k, v in g.items():
                            Expectation_g_2[k] = 0.1 * v * v + 0.9 * Expectation_g_2[k]
                            iter_param_delta[k] = torch.sqrt((iter_param_s[k] + 1e-6) / (Expectation_g_2[k] + 1e-6)) * v
                            iter_param_s[k] = 0.1 * iter_param_delta[k] * iter_param_delta[k] + 0.9 * iter_param_s[k]
                        for k, v in w_local.items():
                            v = v - iter_param_delta[k]
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            w_local_previous = w_local.copy()
                            Expectation_g_2_used = Expectation_g_2.copy()
                            iter_param_s_used = iter_param_s.copy()
                        net_glob.load_state_dict(w_local)
                        if aa == int(args.local_ep):
                            break


                    elif args.optimizer == "RMSPROP":
                        optimizer = torch.optim.RMSprop(net_glob.parameters(), lr=args.lr)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()
                        loss = loss_func(log_probs, labels)
                        loss.backward()
                        g = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)
                        for k, v in g.items():
                            Expectation_g_2[k] = 0.1 * v * v + 0.9 * Expectation_g_2[k]
                        delta_w = dict()
                        for k, v in Expectation_g_2.items():
                            delta_w[k] = g[k] / torch.sqrt(v + 1e-8)
                        for k, v in w_local.items():
                            v = v - args.lr * delta_w[k]
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            Expectation_g_2_used = Expectation_g_2.copy()
                            w_local_previous = w_local.copy()
                        net_glob.load_state_dict(w_local)
                        if aa == int(args.local_ep):
                            break

                    elif args.optimizer == "Adam":
                        optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()
                        loss = loss_func(log_probs, labels)
                        loss.backward()
                        g = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)
                        for k, v in g.items():
                            iter_m[k] = 0.9*iter_m[k]+0.1*g[k]
                            iter_v[k] = 0.999*iter_v[k]+0.001*g[k]*g[k]
                        hat_m = dict()
                        hat_v = dict()
                        for k, v in iter_m.items():
                            hat_m[k]=iter_m[k]/(1-0.9**aa)
                            hat_v[k]=iter_v[k]/(1-0.999**aa)
                        for k, v in w_local.items():
                            v = v - args.lr * hat_m[k]/(torch.sqrt(hat_v[k])+1e-8)
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            iter_m_used = iter_m.copy()
                            iter_v_used = iter_v.copy()
                            w_local_previous = w_local.copy()
                        net_glob.load_state_dict(w_local)
                        if aa == int(args.local_ep):
                            break

            if aa == int(args.local_ep):
                break


        s = dict()
        sigma = dict()
        delta_w_sum = dict()

        delta_w_sum_store = torch.empty(1).to(args.device)
        s_store = torch.empty(1).to(args.device)

        for k, v in w_local.items():
            delta_w_sum[k] = w_local_0[k] - v


        for idx_key in key:



            w_local_shape = w_local[idx_key].shape


            s[idx_key] = torch.zeros(w_local_shape).to(args.device_c)

            if args.optimizer == "RMSPROP":
                s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key] - args.lr * g_used[idx_key] /
                                                   torch.sqrt(0.9 * Expectation_g_2_used[idx_key] +
                                                              0.1 * g_used[idx_key] * g_used[idx_key] + 1e-8) -
                                                   w_local_0[idx_key])

            elif args.optimizer == "SGD":
                s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key] - args.lr * g_used[idx_key] - w_local_0[idx_key])

            elif args.optimizer == "Momentum":
                s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key]
                                                   - (0.9 * iter_param_used[idx_key] + args.lr * g_used[idx_key])
                                                   - w_local_0[idx_key])
            elif args.optimizer == "Adadelta":
                s[idx_key] = args.beta * torch.abs(
                    w_local_previous[idx_key] - g_used[idx_key] * torch.sqrt(
                        (iter_param_s_used[idx_key] + 1e-6) / (
                                0.9 * Expectation_g_2_used[idx_key] + 0.1 *
                                g_used[idx_key] * g_used[idx_key] + 1e-8)) - w_local_0[idx_key])

            elif args.optimizer == "Adam":
                s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key] - args.lr*(0.9*iter_m_used[idx_key]+0.1*g_used[idx_key])/((1-0.9*aa)*(1e-8+torch.sqrt((0.999*iter_v_used[idx_key]+0.001*g_used[idx_key]*g_used[idx_key])/(1-0.999**aa)))) - w_local_0[idx_key])

            m = 1
            for product in w_local[idx_key].size():
                m = m * product


            sigma[idx_key] = s[idx_key] * sigma_0 * (m ** 0.5)


            delta_w_sum[idx_key] = torch.min(torch.max(delta_w_sum[idx_key], -s[idx_key]), s[idx_key])


            delta_w_sum[idx_key] = delta_w_sum[idx_key] + \
                                   torch.normal(0, sigma[idx_key]/batch_size).to(args.device_c)


            w_local[idx_key] = w_local_0[idx_key] - delta_w_sum[idx_key]


        if idx in Z_k_idxs_users:
            fed_avg_collect.add(w_local, client_FL_weights[idx])
            L[idx] += 1

            delta = torch.zeros(alpha.shape)

            for i in range(63):
                delta[i] = torch.exp(-(alpha[i] - 1) * (epsilon - (L[idx] + 1) * alpha[i] / (2 * sigma_0 ** 2)))

            delta_locals[idx] = torch.min(delta)

    delta_c = torch.max(delta_locals)

    w_global = fed_avg_collect.get_result()

    fed_avg_collect.clear()

    net_glob.load_state_dict(w_global)

    print('rounds:', k_iter)

    net_glob.eval()
    final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
    final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(final_acc_train))
    print("Testing accuracy: {:.2f}".format(final_acc_test))

    acc_test_repro.append(final_acc_test)
    acc_train_repro.append(final_acc_train)

    if delta_c > args.delta:
        break


print("Training accuracy epsilon{}_frac{}_lr{}_nl{}_gamma{}_ls{}_an{}_users{}: {:.2f}".format(
    args.epsilon, args.frac, args.lr, args.nl, args.beta, args.ls, args.an, args.num_users,
    final_acc_train))
print("Testing accuracy epsilon{}_frac{}_lr{}_nl{}_gamma{}_ls{}_an{}_users{}: {:.2f}".format(
    args.epsilon, args.frac, args.lr, args.nl, args.beta, args.ls, args.an, args.num_users,
    final_acc_test))
print('Finish!')
