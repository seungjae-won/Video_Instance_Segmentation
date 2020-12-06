import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from utill import *
from utill import save_model

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

class Train:
    
    def __init__(self, args):

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        
        self.data_dir = args.data_dir
        self.ckpt_dir = args.ckpt_dir
        self.log_dir = args.log_dir
        self.result_dir = args.result_dir

        self.mode = args.mode
        self.image_type = args.image_type
        self.train_continue = args.train_continue
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        
    def UNet(self):
        if self.mode == 'train':
            transform = transforms.Compose([Normalization(mean=0.5, std=0.5, mode='train'), ToTensor()])

            dataset_train = Dataset(mode = self.mode, data_dir=self.data_dir, image_type = self.image_type, transform=transform)
            loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

            # dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
            # loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

            # 그밖에 부수적인 variables 설정하기
            num_data_train = len(dataset_train)
            # num_data_val = len(dataset_val)

            num_batch_train = np.ceil(num_data_train / self.batch_size)
            # num_batch_val = np.ceil(num_data_val / batch_size)
            
        elif self.mode == 'test':
            transform = transforms.Compose([Normalization(mean=0.5, std=0.5, mode='test'), ToTensor()])

            dataset_test = Dataset(mode = self.mode, data_dir=self.data_dir, image_type = self.image_type, transform=transform)
            loader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=8)

            # 그밖에 부수적인 variables 설정하기
            num_data_test = len(dataset_test)

            num_batch_test = np.ceil(num_data_test / self.batch_size)
        

        fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        fn_denorm = lambda x, mean, std: (x * std) + mean
        fn_class = lambda x: 1.0 * (x > 0.5)
        
        
        net = UNet().to(self.device)
        
        criterion = torch.nn.MSELoss().to(self.device)
            
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        

        writer_train = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))

            
            
        if self.mode == 'train':
            if self.train_continue == "on":
                net, optimizer = load_model(ckpt_dir=self.ckpt_dir, net=net, optim=optimizer)

            for epoch in range(1, self.num_epoch + 1):
                net.train()
                loss_arr = []

                for batch, data in enumerate(loader_train, 1):
                    # forward pass
                    label = data['label'].to(self.device)
                    input = data['input'].to(self.device)

                    output = net(input)

                    # backward pass
                    optimizer.zero_grad()

                    loss = criterion(output, label)
                    loss.backward()

                    optimizer.step()

                    # 손실함수 계산
                    loss_arr += [loss.item()]

                    

                    # Tensorboard 저장하기
                    label = fn_tonumpy(label)
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    output = fn_tonumpy(fn_class(output))

                    writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                    writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                    writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

                writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
                
                print("TRAIN: EPOCH %04d / %04d |  LOSS %.4f" %(epoch, self.num_epoch, np.mean(loss_arr)))
                
                if epoch % 20 == 0:
                    save_model(ckpt_dir=self.ckpt_dir, net=net, optim=optimizer, epoch=0)
                    
            writer_train.close()


        # TEST MODE
        elif self.mode == 'test':
            net, optimizer = load_model(ckpt_dir=self.ckpt_dir, net=net, optim=optimizer)

            with torch.no_grad():
                net.eval()
                loss_arr = []
                id = 1
                for batch, data in enumerate(loader_test, 1):
                    # forward pass
                    input = data['input'].to(self.device)

                    output = net(input)

                    # 손실함수 계산하기
                    #loss = criterion(output, label)

                    #loss_arr += [loss.item()]

                    #print("TEST: BATCH %04d / %04d | " %
                    #    (batch, num_batch_test))

                    # Tensorboard 저장하기
                    output = fn_tonumpy(fn_class(output))
                    
                    for j in range(input.shape[0]):
                        if id == 800:
                            id = 2350
                        print(id)
                        #plt.imsave(os.path.join(self.result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                        #plt.imsave(os.path.join(self.result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                        plt.imsave(os.path.join(self.result_dir, 'png', 'gt%06d.png' % id), output[j].squeeze(), cmap='gray')
                        id+=1
                        # np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                        # np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                        # np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

            print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_arr)))