import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
from tqdm import tqdm
from sam import  SAM  #参数设置：
from bypass_bn import enable_running_stats, disable_running_stats

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

# 数据文件命名规则：时间+算法名alg+数据分布partition+epochs+抽样比例sample（SAM相关的命名未设置，有SAM的在logdir命名中说明）
for ep in [15]:
    #simple-cnn，fmnist，epochs的影响（全采样/部分采样）
    def get_args():
        parser = argparse.ArgumentParser()
        #基础参数
        parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')   # ******************
        parser.add_argument('--dataset', type=str, default='svhn', help='dataset used for training')  # ******************
        parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')  # ******************
        parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')  # ******************
        parser.add_argument('--epochs', type=int, default=ep, help='number of local epochs')  # ******************
        parser.add_argument('--n_parties', type=int, default=100,  help='number of workers in a distributed cluster')  # ******************
        parser.add_argument('--alg', type=str, default='fedavg',
                                help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')  # ******************
        parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')   # ******************
        parser.add_argument('--beta', type=float, default=0.3, help='The parameter for the dirichlet distribution for data partitioning')  # 迪利克雷分布参数
        parser.add_argument('--sample', type=float, default=1.0, help='Sample ratio for each communication round')  # ******************

        #for SAM
        parser.add_argument("--use_SAM", default=False, type=bool, help="True if you want to use SAM.")
        parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
        parser.add_argument('--rho_SAM', type=float, default=0.1, help='Parameter for SAM')  #0.05，0.1，0.7

        #for FedProx
        parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')

        #for 优化器
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
        parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')  # ******************
        parser.add_argument('--reg', type=float, default=1e-5, help="weight_decay-L2 regularization strength")
        parser.add_argument('--rho', type=float, default=0, help='Parameter-momentum controlling the momentum SGD')

        #for MOON
        parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
        parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
        parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
        parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')

        parser.add_argument('--is_same_initial', type=int, default=1,help='Whether initial all the models with the same parameters in fedavg')
        parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
        parser.add_argument('--dropout_p', type=float, required=False, default=0.0,help="Dropout probability. Default=0.0")
        parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
        parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')

        parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
        parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
        parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
        parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
        parser.add_argument('--logdir', type=str, required=False, default="./logs_SVHN_fedavg_IID/", help='Log directory path')
        parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')



        args = parser.parse_args()
        return args



    def init_nets(net_configs, dropout_p, n_parties, args):
    #初始化各客户端的网络模型
    #dropout_p：Dropout的概率
        nets = {net_i: None for net_i in range(n_parties)}

        if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
            n_classes = 10
        elif args.dataset == 'celeba':
            n_classes = 2
        elif args.dataset == 'cifar100':
            n_classes = 100
        elif args.dataset == 'tinyimagenet':
            n_classes = 200
        elif args.dataset == 'femnist':
            n_classes = 62
        elif args.dataset == 'emnist':
            n_classes = 47
        elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
            n_classes = 2
        if args.use_projection_head:  #for moon
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            if args.alg == 'moon':
                add = ""
                if "mnist" in args.dataset and args.model == "simple-cnn":
                    add = "-mnist"
                for net_i in range(n_parties):
                    net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                    nets[net_i] = net
            else:
                for net_i in range(n_parties):
                    if args.dataset == "generated":
                        net = PerceptronModel()
                    elif args.model == "mlp":
                        if args.dataset == 'covtype':
                            input_size = 54
                            output_size = 2
                            hidden_sizes = [32,16,8]
                        elif args.dataset == 'a9a':
                            input_size = 123
                            output_size = 2
                            hidden_sizes = [32,16,8]
                        elif args.dataset == 'rcv1':
                            input_size = 47236
                            output_size = 2
                            hidden_sizes = [32,16,8]
                        elif args.dataset == 'SUSY':
                            input_size = 18
                            output_size = 2
                            hidden_sizes = [16,8]
                        net = FcNet(input_size, hidden_sizes, output_size, dropout_p)

                    elif args.model == "vgg":
                        net = vgg11()
                    # ******************
                    elif args.model == "simple-cnn":
                        if args.dataset in ("cifar10", "cinic10", "svhn"):
                            net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                        elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                            net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                        elif args.dataset == 'celeba':
                            net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                    elif args.model == "vgg-9":
                        if args.dataset in ("mnist", 'femnist'):
                            net = ModerateCNNMNIST()
                        elif args.dataset in ("cifar10", "cinic10", "svhn"):
                            # print("in moderate cnn")
                            net = ModerateCNN()
                        elif args.dataset == 'celeba':
                            net = ModerateCNN(output_dim=2)
                    elif args.model == "resnet":
                        net = ResNet18_cifar10()
                    elif args.model == "vgg16":
                        net = vgg16()
                    else:
                        print("not supported yet")
                        exit(1)
                    nets[net_i] = net

        model_meta_data = []
        layer_type = []
        for (k, v) in nets[0].state_dict().items():
            model_meta_data.append(v.shape)   #网络权重的形状，列表
            layer_type.append(k)   #网络层的类型信息，列表
        return nets, model_meta_data, layer_type   #输出中的nets是字典，键是参与方的索引，值是对应的网络实例；


    def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu",args_use_SAM=False):
    #主要步骤包括：计算初始的训练和测试准确率，设置优化器和损失函数，进行多轮训练（每轮包括前向传播、计算损失、反向传播和参数更新），
    #输出：训练后在训练数据集与测试数据集上的准确率

        logger.info('Training network %s' % str(net_id))

        train_acc = compute_accuracy(net, train_dataloader, device=device)   #默认 get_confusion_matrix=False,moon_model=False
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

        if args_use_SAM == False:
            if args_optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
            elif args_optimizer == 'amsgrad':
            #weight_decay是在优化器中用于实现权重衰减（也称为L2正则化）的参数。权重衰减是一种正则化技术，用于防止模型过拟合。
            #在每次参数更新时，权重衰减都会将一部分的权重值加到损失函数中。这意味着模型的权重将会逐渐减小
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                                       amsgrad=True)
            elif args_optimizer == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
            criterion = nn.CrossEntropyLoss().to(device)
            #filter(lambda p: p.requires_grad, net.parameters())：这部分代码过滤出网络中需要进行梯度更新的参数。
            #net.parameters()返回网络的所有参数，lambda p: p.requires_grad是一个函数，用于检查每个参数是否需要进行梯度更新（即requires_grad=True）


            cnt = 0
            if type(train_dataloader) == type([1]):
                pass
            else:
                train_dataloader = [train_dataloader]

            #writer = SummaryWriter()

            for epoch in range(epochs):
                epoch_loss_collector = []
                for tmp in train_dataloader:
                    for batch_idx, (x, target) in enumerate(tmp):
                        x, target = x.to(device), target.to(device)

                        optimizer.zero_grad()
                        x.requires_grad = True
                        target.requires_grad = False
                        target = target.long()
                        #在PyTorch中，long()函数用于将数据转换为长整型（Long）张量。
                        #在处理分类问题时，标签数据通常需要为长整型，因为损失函数（如交叉熵损失CrossEntropyLoss）需要接收长整型的标签数据。
                        out = net(x)
                        loss = criterion(out, target)

                        loss.backward()
                        optimizer.step()

                        cnt += 1
                        epoch_loss_collector.append(loss.item())
                        #.item()方法用于获取一个标量张量（scalar tensor）的Python数值，以便进行打印、记录或其他非张量计算

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

                #train_acc = compute_accuracy(net, train_dataloader, device=device)
                #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

                #writer.add_scalar('Accuracy/train', train_acc, epoch)
                #writer.add_scalar('Accuracy/test', test_acc, epoch)

                # if epoch % 10 == 0:
                #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
                #     train_acc = compute_accuracy(net, train_dataloader, device=device)
                #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
                #
                #     logger.info('>> Training accuracy: %f' % train_acc)
                #     logger.info('>> Test accuracy: %f' % test_acc)

            train_acc = compute_accuracy(net, train_dataloader, device=device)  #此处net已经训练过了
            test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

            net.to('cpu')
            #在完成GPU上的模型训练后使用，以便将模型移动到CPU进行推理，或者在保存模型前将模型移动到CPU

            logger.info(' ** Training complete **')
            return train_acc, test_acc
        else:
            # print("USE_SAM:",args_use_SAM)
            base_optimizer = optim.SGD
            optimizer = SAM(net.parameters(), base_optimizer, rho=args.rho_SAM, adaptive=args.adaptive,
                            lr=lr, momentum=args.rho, weight_decay=args.reg)
            criterion = nn.CrossEntropyLoss().to(device)

            cnt = 0
            if type(train_dataloader) == type([1]):
                pass
            else:
                train_dataloader = [train_dataloader]

            for epoch in range(epochs):
                epoch_loss_collector = []
                for tmp in train_dataloader:
                    for batch_idx, (x, target) in enumerate(tmp):
                        x, target = x.to(device), target.to(device)
                        x.requires_grad = True
                        target.requires_grad = False
                        target = target.long()
                        #在PyTorch中，long()函数用于将数据转换为长整型（Long）张量。
                        #在处理分类问题时，标签数据通常需要为长整型，因为损失函数（如交叉熵损失CrossEntropyLoss）需要接收长整型的标签数据。
                        enable_running_stats(net)
                        out = net(x)
                        loss = criterion(out, target)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        disable_running_stats(net)
                        criterion(net(x), target).backward()
                        optimizer.second_step(zero_grad=True)
                        cnt += 1
                        epoch_loss_collector.append(loss.item())
                        #.item()方法用于获取一个标量张量（scalar tensor）的Python数值，以便进行打印、记录或其他非张量计算

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

                #train_acc = compute_accuracy(net, train_dataloader, device=device)
                #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

                #writer.add_scalar('Accuracy/train', train_acc, epoch)
                #writer.add_scalar('Accuracy/test', test_acc, epoch)

                # if epoch % 10 == 0:
                #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
                #     train_acc = compute_accuracy(net, train_dataloader, device=device)
                #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
                #
                #     logger.info('>> Training accuracy: %f' % train_acc)
                #     logger.info('>> Test accuracy: %f' % test_acc)

            train_acc = compute_accuracy(net, train_dataloader, device=device)  #此处net已经训练过了
            test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

            net.to('cpu')
            #在完成GPU上的模型训练后使用，以便将模型移动到CPU进行推理，或者在保存模型前将模型移动到CPU

            logger.info(' ** Training complete **')
            return train_acc, test_acc

    def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    #net_dataidx_map：一个字典，键是客户端的ID，值是对应网络需要处理的数据的索引
    #返回训练后各客户端的模型列表
        avg_acc = 0.0

        for net_id, net in nets.items():  #对每个被选中的客户端进行模型训练

            '''
            在Python中，for循环中的变量net是nets字典中的一个副本，而不是引用。因此，如果你修改net，
            它不会影响nets字典中的原始对象。但是，如果你的net是一个可变对象（例如，列表或字典）
            并且你修改了这个对象的内容（例如，添加、删除或修改元素），那么这些更改将反映在nets字典中的原始对象上。
            在你给出的代码中，net是一个神经网络模型。如果你修改了模型的参数（例如，通过训练），
            那么这些更改将反映在nets字典中的原始模型上，因为模型的参数是存储在模型对象内部的可变对象
            '''
            if net_id not in selected:
                continue
            dataidxs = net_dataidx_map[net_id]

            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            # move the model to cuda device:
            net.to(device)

            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
                '''
                如果当前网络的ID(net_id)等于所有网络的最大ID(args.n_parties - 1)，那么将噪声级别(noise_level)设置为0，否则噪声级别为预设的args.noise。
                这可能是因为在某些分布式学习或联邦学习的场景中，最后一个网络可能被设计为没有噪声，或者说是一个“干净”的网络。
                '''

            if args.noise_type == 'space':
                #注意test_bs设定为32
                #输出的前两者：经过DataLoader后的训练与测试数据，后两者是未经过DataLoader的训练和测试数据
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
            n_epoch = args.epochs


            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device,args_use_SAM=args_use_SAM)  #训练
            logger.info("net %d final test acc %f" % (net_id, testacc))
            avg_acc += testacc
            # saving the trained models here
            # save_model(net, net_id, args)
            # else:
            #     load_model(net, net_id, device=device)
        avg_acc /= len(selected)
        if args.alg == 'local_training':
            #local_training
            logger.info("avg test acc %f" % avg_acc)

        nets_list = list(nets.values())
        return nets_list  #返回训练后各客户端的模型列表


    def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, device="cpu"):
    #FedProx算法单个客户端的训练过程，输入增加了global_net、mu
    #输出：训练后在训练数据集与测试数据集上的准确率
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))

        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


        if args_optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        elif args_optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                                   amsgrad=True)
        elif args_optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

        criterion = nn.CrossEntropyLoss().to(device)

        cnt = 0
        # mu = 0.001
        global_weight_collector = list(global_net.to(device).parameters())

        for epoch in range(epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                #for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                loss += fed_prox_reg


                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

            # if epoch % 10 == 0:
            #     train_acc = compute_accuracy(net, train_dataloader, device=device)
            #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
            #
            #     logger.info('>> Training accuracy: %f' % train_acc)
            #     logger.info('>> Test accuracy: %f' % test_acc)

        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)

        net.to('cpu')
        logger.info(' ** Training complete **')
        return train_acc, test_acc

    def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    #注意：c_global, c_local, global_model
    #输出：训练后在训练数据集与测试数据集上的准确率、更新前后c_local的变化值
        logger.info('Training network %s' % str(net_id))

        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

        if args_optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        elif args_optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                                   amsgrad=True)
        elif args_optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
        criterion = nn.CrossEntropyLoss().to(device)

        cnt = 0
        if type(train_dataloader) == type([1]):
            pass
        else:
            train_dataloader = [train_dataloader]

        #writer = SummaryWriter()

        c_global_para = c_global.state_dict()
        c_local_para = c_local.state_dict()

        for epoch in range(epochs):
            epoch_loss_collector = []
            for tmp in train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    out = net(x)
                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    net_para = net.state_dict()
                    for key in net_para:  #scaffold本地模型的更新
                        net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                    net.load_state_dict(net_para)

                    cnt += 1
                    epoch_loss_collector.append(loss.item())


            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        c_new_para = c_local.state_dict()
        c_delta_para = copy.deepcopy(c_local.state_dict())
        global_model_para = global_model.state_dict()
        net_para = net.state_dict()
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
            #更新c_local
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
            #计算更新前后c_local的变化值
        c_local.load_state_dict(c_new_para)


        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)

        net.to('cpu')
        logger.info(' ** Training complete **')
        return train_acc, test_acc, c_delta_para

    def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    #注意： global_model
    #输出：训练后在训练数据集与测试数据集上的准确率、a_i代表训练过程中的迭代次数、norm_grad：字典，存储了全局模型和本地模型之间的参数差（除以a_i）
        logger.info('Training network %s' % str(net_id))

        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


        #优化器只有SGD
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
        criterion = nn.CrossEntropyLoss().to(device)

        if type(train_dataloader) == type([1]):
            pass
        else:
            train_dataloader = [train_dataloader]

        #writer = SummaryWriter()


        tau = 0

        for epoch in range(epochs):
            epoch_loss_collector = []
            for tmp in train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    out = net(x)
                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    tau = tau + 1

                    epoch_loss_collector.append(loss.item())


            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


        a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho) #对于tau是单调递增的
        global_model_para = global_model.state_dict()
        net_para = net.state_dict()
        norm_grad = copy.deepcopy(global_model.state_dict())
        for key in norm_grad:
            #norm_grad是一个字典，存储了全局模型和本地模型之间的参数差的归一化值。
            #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
            norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)

        net.to('cpu')
        logger.info(' ** Training complete **')
        return train_acc, test_acc, a_i, norm_grad


    def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                          round, device="cpu"):
    #注意：global_net, previous_nets, temperature,round
    #输出：训练后在训练数据集与测试数据集上的准确率
        logger.info('Training network %s' % str(net_id))

        train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device)

        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

        # conloss = ContrastiveLoss(temperature)

        if args_optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        elif args_optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                                   amsgrad=True)
        elif args_optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                                  weight_decay=args.reg)

        criterion = nn.CrossEntropyLoss().to(device)
        # global_net.to(device)

        if args.loss != 'l2norm':
            for previous_net in previous_nets:
                previous_net.to(device)
        global_w = global_net.state_dict()
        # oppsi_nets = copy.deepcopy(previous_nets)
        # for net_id, oppsi_net in enumerate(oppsi_nets):
        #     oppsi_w = oppsi_net.state_dict()
        #     prev_w = previous_nets[net_id].state_dict()
        #     for key in oppsi_w:
        #         oppsi_w[key] = 2*global_w[key] - prev_w[key]
        #     oppsi_nets.load_state_dict(oppsi_w)
        cnt = 0
        cos=torch.nn.CosineSimilarity(dim=-1).to(device)#计算两个向量之间的余弦相似度，dim=-1示计算最后一个维度上的余弦相似度
        # mu = 0.001

        for epoch in range(epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                _, pro1, out = net(x)
                _, pro2, _ = global_net(x)
                if args.loss == 'l2norm':
                    loss2 = mu * torch.mean(torch.norm(pro2-pro1, dim=1))

                elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                    posi = cos(pro1, pro2)
                    logits = posi.reshape(-1,1)  ##将logits张量转换为列向量，本地模型与全局模型的差异要尽可能小

                    for previous_net in previous_nets:
                        previous_net.to(device)
                        _, pro3, _ = previous_net(x)
                        nega = cos(pro1, pro3)   #上一轮本地模型与当前本地模型的差异要尽可能大
                        logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                        #将nega张量转换为列向量，并将其添加到logits张量的第1维上
                        #nega.reshape(-1,1)是将nega张量重新塑形为一个列向量。-1在reshape函数中表示该维度的大小会自动计算，以保证总的元素数量不变
                        # previous_net.to('cpu')

                    logits /= temperature
                    labels = torch.zeros(x.size(0)).to(device).long()  #.size()是一个方法，用于获取张量（Tensor）的形状。
                                                                       #如果x是一个形状为(batch_size, channels, height, width)的张量（这是一个常见的形状，用于存储一批图像数据），那么x.size(0)就会返回batch_size
                    #创建一个全零的一维张量，其长度等于x的批次大小，并将其转换为长整型（long）并移动到指定的设备上
                    # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                    loss2 = mu * criterion(logits, labels)  #对比损失？？

                if args.loss == 'only_contrastive':
                    loss = loss2
                else:
                    loss1 = criterion(out, target)
                    loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


        if args.loss != 'l2norm':
            for previous_net in previous_nets:
                previous_net.to('cpu')
        train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
        net.to('cpu')
        logger.info(' ** Training complete **')
        return train_acc, test_acc


    def view_image(train_dataloader):
        for (x, target) in train_dataloader:
            np.save("img.npy", x)
            print(x.shape)
            exit(0)
        # 使用exit(0)结束程序。这意味着这个函数只会处理train_dataloader的第一个批次，然后就会停止。
        # 如果你想处理所有批次，你需要移除这行代码。


    #上面定义了一个客户端是怎样训练的，并输出训练后在训练数据集与测试数据集上的准确率以及一些算法特有的参数
    #接下来，定义的函数用于让选中的所有客户端参与训练，并输出得到的客户端模型列表




    def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
         #注意：global_model
        avg_acc = 0.0

        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            dataidxs = net_dataidx_map[net_id]

            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            # move the model to cuda device:
            net.to(device)

            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
            n_epoch = args.epochs

            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args.mu, device=device)
            logger.info("net %d final test acc %f" % (net_id, testacc))
            avg_acc += testacc
        avg_acc /= len(selected)
        if args.alg == 'local_training':
            logger.info("avg test acc %f" % avg_acc)

        nets_list = list(nets.values())
        return nets_list

    def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu"):
        #注意：global_model, c_nets, c_global
        avg_acc = 0.0

        total_delta = copy.deepcopy(global_model.state_dict())
        for key in total_delta:
            total_delta[key] = 0.0
            #创建一个与global_model的状态字典（state_dict）结构相同，但所有值都初始化为0的新字典total_delta
        c_global.to(device)
        global_model.to(device)
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            dataidxs = net_dataidx_map[net_id]

            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            # move the model to cuda device:
            net.to(device)

            c_nets[net_id].to(device)

            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
            n_epoch = args.epochs


            trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

            c_nets[net_id].to('cpu')
            for key in total_delta:
                total_delta[key] += c_delta_para[key]


            logger.info("net %d final test acc %f" % (net_id, testacc))
            avg_acc += testacc
        for key in total_delta:
            total_delta[key] /= args.n_parties
        c_global_para = c_global.state_dict()
        for key in c_global_para:
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
            else:
                #print(c_global_para[key].type())
                c_global_para[key] += total_delta[key]
        c_global.load_state_dict(c_global_para)
        #更新了c_global

        avg_acc /= len(selected)
        if args.alg == 'local_training':
            logger.info("avg test acc %f" % avg_acc)

        nets_list = list(nets.values())
        return nets_list

    def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
        #注意：global_model
        avg_acc = 0.0

        a_list = []
        d_list = []
        n_list = []
        global_model.to(device)
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            dataidxs = net_dataidx_map[net_id]

            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            # move the model to cuda device:
            net.to(device)

            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
            n_epoch = args.epochs


            trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

            a_list.append(a_i)
            d_list.append(d_i)
            n_i = len(train_dl_local.dataset)
            n_list.append(n_i)
            logger.info("net %d final test acc %f" % (net_id, testacc))
            avg_acc += testacc


        avg_acc /= len(selected)
        if args.alg == 'local_training':
            logger.info("avg test acc %f" % avg_acc)

        nets_list = list(nets.values())
        return nets_list, a_list, d_list, n_list #输出：客户端模型列表，客户端局部迭代次数列表、客户端模型更新前后模型差异列表（已经除过迭代次数了）、客户端数据量大小列表

    def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model = None, prev_model_pool = None, round=None, device="cpu"):
        #注意：global_model , prev_model_pool , round
        avg_acc = 0.0
        global_model.to(device)
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            dataidxs = net_dataidx_map[net_id]

            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            net.to(device)

            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
            n_epoch = args.epochs

            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
                #找到该客户端所有的历史模型
            trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device)
            logger.info("net %d final test acc %f" % (net_id, testacc))
            avg_acc += testacc

        avg_acc /= len(selected)
        if args.alg == 'local_training':
            logger.info("avg test acc %f" % avg_acc)
        global_model.to('cpu')
        nets_list = list(nets.values())
        return nets_list



    def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    #输出各个客户端拥有的数据索引
        seed = init_seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
            dataset, datadir, logdir, partition, n_parties, beta=beta)
        #训练数据（X_train, y_train）、测试数据（X_test, y_test）、各个客户端拥有的数据索引net_dataidx_map、各参与方（网络节点）拥有的数据的类别统计信息traindata_cls_counts
        return net_dataidx_map

    if __name__ == '__main__':
        # torch.set_printoptions(profile="full")
        args = get_args()
        args_use_SAM = args.use_SAM
        mkdirs(args.logdir)
        mkdirs(args.modeldir) #创建目录
        # 数据文件命名规则：时间+算法名alg+数据分布partition+epochs+抽样比例sample
        name_alg=str(args.alg)
        name_partition = str(args.partition)
        name_epochs = str(args.epochs)
        name_sample = str(args.sample)
        if args.log_file_name is None:

            argument_path0='arguments-%s-' % datetime.datetime.now().strftime("%Y-%m-%d-%H_%M-%S")
            argument_path =argument_path0+'-'+name_alg+'-'+name_partition+'-'+'epoch='+name_epochs+'-'+name_sample+'.json'
        else:
            argument_path=args.log_file_name+'.json'
        with open(os.path.join(args.logdir, argument_path), 'w') as f:
            json.dump(str(args), f)
        device = torch.device(args.device)
        # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
        # logging.info("test")
        for handler in logging.root.handlers[:]:  #移除所有已经存在的日志处理器。这是为了避免日志信息被重复记录
            logging.root.removeHandler(handler)

        if args.log_file_name is None:#设置日志文件的名称
            log_file_name0 = 'log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H_%M-%S"))

            args.log_file_name =log_file_name0+'-'+name_alg+'-'+name_partition+'-'+'epoch='+name_epochs+'-'+name_sample+'.log'
        log_path=args.log_file_name+'.log'
        logging.basicConfig(   #初始化日志系统的基本配置
            filename=os.path.join(args.logdir, log_path),
            # filename='/home/qinbin/test.log',
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

        logger = logging.getLogger()  #获取一个日志器对象
        logger.setLevel(logging.DEBUG)
        logger.info(device)

        seed = args.init_seed
        logger.info("#" * 100)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        logger.info("Partitioning data")
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
            args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
        #训练数据（X_train, y_train）、测试数据（X_test, y_test）、各个客户端拥有的数据索引、各参与方（网络节点）拥有的数据的类别统计信息

        n_classes = len(np.unique(y_train))#找出数组中的唯一元素，并返回已排序的结果，此处返回了类别数

        train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                            args.datadir,
                                                                                            args.batch_size,
                                                                                            32)
        #输出：train_dl_global, test_dl_global：训练数据集与测试数据集的DataLoader，train_ds_global：训练数据集中的图片与标签（未设置数据索引）， test_ds_global：测试数据集的图片与标签

        print("len train_dl_global:", len(train_ds_global))


        data_size = len(test_ds_global)

        # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

        train_all_in_list = []
        test_all_in_list = []
        if args.noise > 0:
            #为每个参与方（party）创建本地的训练和测试数据加载器（DataLoader）。如果存在噪声，则会根据噪声级别和类型对数据进行处理。此处的添加噪声用于集中式的训练。其它算法在训练的函数中自定义了添加噪声
            for party_id in range(args.n_parties):
                dataidxs = net_dataidx_map[party_id]

                noise_level = args.noise
                if party_id == args.n_parties - 1:
                    noise_level = 0

                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
                else:
                    noise_level = args.noise / (args.n_parties - 1) * party_id
                    train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
                train_all_in_list.append(train_ds_local)
                test_all_in_list.append(test_ds_local)
                #每个参与方的本地训练和测试数据集添加到train_all_in_list和test_all_in_list列表中
            train_all_in_ds = data.ConcatDataset(train_all_in_list)
            train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
            test_all_in_ds = data.ConcatDataset(test_all_in_list)
            test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)
            #使用data.ConcatDataset将所有参与方的训练和测试数据集合并，创建全局的训练和测试数据集


        if args.alg == 'fedavg':
            logger.info("Initializing nets")
            #客户端模型
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            #全局模型
            global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
            global_model = global_models[0]

            #将全局模型参数复制给各个客户端模型
            global_para = global_model.state_dict()
            if args.is_same_initial:
                for net_id, net in nets.items():
                    net.load_state_dict(global_para)

            #开始训练
            for round in tqdm(range(args.comm_round)):
                logger.info("in comm round:" + str(round))

                #每轮随机选择一定比例的客户端
                arr = np.arange(args.n_parties)
                np.random.shuffle(arr)
                selected = arr[:int(args.n_parties * args.sample)]

                global_para = global_model.state_dict()

                #将全局模型发送给客户端
                if round == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_para)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)


                #客户端进行训练
                local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)
                # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

                # 更新全局模型
                #计算客户端模型的权重
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                for idx in range(len(selected)):
                    net_para = nets[selected[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_para[key] += net_para[key] * fed_avg_freqs[idx]
                global_model.load_state_dict(global_para)

                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(test_dl_global))

                global_model.to(device)
                train_acc = compute_accuracy(global_model, train_dl_global, device=device)
                test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

                #输出在整体训练数据与测试数据集上的准确率
                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

        elif args.alg == 'fedprox':
            logger.info("Initializing nets")
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
            global_model = global_models[0]

            global_para = global_model.state_dict()

            if args.is_same_initial:
                for net_id, net in nets.items():
                    net.load_state_dict(global_para)

            for round in range(args.comm_round):
                logger.info("in comm round:" + str(round))

                arr = np.arange(args.n_parties)
                np.random.shuffle(arr)
                selected = arr[:int(args.n_parties * args.sample)]

                global_para = global_model.state_dict()
                if round == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_para)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)

                local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
                global_model.to('cpu')

                # update global model
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                for idx in range(len(selected)):
                    net_para = nets[selected[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_para[key] += net_para[key] * fed_avg_freqs[idx]
                global_model.load_state_dict(global_para)


                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(test_dl_global))


                global_model.to(device)
                train_acc = compute_accuracy(global_model, train_dl_global, device=device)
                test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)


                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

        elif args.alg == 'scaffold':
            logger.info("Initializing nets")
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
            global_model = global_models[0]

            c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
            c_global = c_globals[0]
            c_global_para = c_global.state_dict()
            for net_id, net in c_nets.items():
                net.load_state_dict(c_global_para)

            global_para = global_model.state_dict()
            if args.is_same_initial:
                for net_id, net in nets.items():
                    net.load_state_dict(global_para)


            for round in range(args.comm_round):
                logger.info("in comm round:" + str(round))

                arr = np.arange(args.n_parties)
                np.random.shuffle(arr)
                selected = arr[:int(args.n_parties * args.sample)]

                global_para = global_model.state_dict()
                if round == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_para)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)

                local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl_global, device=device)
                # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

                # update global model
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                for idx in range(len(selected)):
                    net_para = nets[selected[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_para[key] += net_para[key] * fed_avg_freqs[idx]
                global_model.load_state_dict(global_para)


                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(test_dl_global))

                global_model.to(device)
                train_acc = compute_accuracy(global_model, train_dl_global, device=device)
                test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

        elif args.alg == 'fednova':
            logger.info("Initializing nets")
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
            global_model = global_models[0]

            d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
            d_total_round = copy.deepcopy(global_model.state_dict())
            for i in range(args.n_parties):
                for key in d_list[i]:
                    d_list[i][key] = 0
            for key in d_total_round:
                d_total_round[key] = 0

            data_sum = 0
            for i in range(args.n_parties):
                data_sum += len(traindata_cls_counts[i])
            portion = []
            for i in range(args.n_parties):
                portion.append(len(traindata_cls_counts[i]) / data_sum)

            global_para = global_model.state_dict()
            if args.is_same_initial:
                for net_id, net in nets.items():
                    net.load_state_dict(global_para)

            for round in range(args.comm_round):
                logger.info("in comm round:" + str(round))

                arr = np.arange(args.n_parties)
                np.random.shuffle(arr)
                selected = arr[:int(args.n_parties * args.sample)]

                global_para = global_model.state_dict()
                if round == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_para)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)

                _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
                total_n = sum(n_list)
                #print("total_n:", total_n)
                d_total_round = copy.deepcopy(global_model.state_dict())
                for key in d_total_round:
                    d_total_round[key] = 0.0

                for i in range(len(selected)):
                    d_para = d_list[i]
                    for key in d_para:
                        #if d_total_round[key].type == 'torch.LongTensor':
                        #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                        #else:
                        d_total_round[key] += d_para[key] * n_list[i] / total_n


                # for i in range(len(selected)):
                #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

                # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

                # update global model
                coeff = 0.0
                for i in range(len(selected)):
                    coeff = coeff + a_list[i] * n_list[i]/total_n

                updated_model = global_model.state_dict()
                for key in updated_model:
                    #print(updated_model[key])
                    if updated_model[key].type() == 'torch.LongTensor':
                        updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                    elif updated_model[key].type() == 'torch.cuda.LongTensor':
                        updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                    else:
                        #print(updated_model[key].type())
                        #print((coeff*d_total_round[key].type()))
                        updated_model[key] -= coeff * d_total_round[key]
                global_model.load_state_dict(updated_model)


                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(test_dl_global))

                global_model.to(device)
                train_acc = compute_accuracy(global_model, train_dl_global, device=device)
                test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)


                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

        elif args.alg == 'moon':
            logger.info("Initializing nets")
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
            global_model = global_models[0]

            global_para = global_model.state_dict()
            if args.is_same_initial:
                for net_id, net in nets.items():
                    net.load_state_dict(global_para)

            old_nets_pool = []
            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False

            for round in range(args.comm_round):
                logger.info("in comm round:" + str(round))

                arr = np.arange(args.n_parties)
                np.random.shuffle(arr)
                selected = arr[:int(args.n_parties * args.sample)]

                global_para = global_model.state_dict()
                if round == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_para)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)

                local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, global_model=global_model,
                                     prev_model_pool=old_nets_pool, round=round, device=device)
                # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

                # update global model
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                for idx in range(len(selected)):
                    net_para = nets[selected[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_para[key] += net_para[key] * fed_avg_freqs[idx]
                global_model.load_state_dict(global_para)

                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(test_dl_global))

                global_model.to(device)
                train_acc = compute_accuracy(global_model, train_dl_global, moon_model=True, device=device)
                test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, moon_model=True, device=device)


                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                if len(old_nets_pool) < 1:
                    old_nets_pool.append(old_nets)
                else:
                    old_nets_pool[0] = old_nets

        elif args.alg == 'local_training':  #各客户端不协作、单独训练
            logger.info("Initializing nets")
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            arr = np.arange(args.n_parties)
            local_train_net(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device)

        elif args.alg == 'all_in':  #集中式的训练
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
            n_epoch = args.epochs
            nets[0].to(device)
            trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

            logger.info("All in test acc: %f" % testacc)


