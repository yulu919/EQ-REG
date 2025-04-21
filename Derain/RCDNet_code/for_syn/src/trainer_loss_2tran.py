import os
import math
from decimal import Decimal
import utility
import IPython
import torch
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as sio 
import model.F_Conv_YL_2tran as Fc
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.S = args.stage
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        # self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch #+ 1
      #  lr = self.scheduler.get_lr()[0]
        lr = self.optimizer.param_groups[0]['lr']
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        loss_Bs_all = 0
        loss_Rs_all = 0
        loss_B_all = 0
        loss_R_all=0
        cnt = 0
        lamda = 1e-8
        print("########lamda:", lamda)
        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            loss_Bs = 0
            loss_Rs = 0
            cnt = cnt+1
            lr, hr = self.prepare(lr, hr)

            n = random.randint(0, 1)
            # print("train pre num:", n)

            lr_rot = self.rot(lr, n)
            hr_rot = self.rot(hr, n)

            lr = torch.cat((lr, lr_rot), dim=0)
            hr = torch.cat((hr, hr_rot), dim=0)

            timer_data.hold()
            timer_model.tic()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            self.model.model.rot_num_in(rot_num=n)

            B0, ListB, ListR = self.model(lr, idx_scale)
            # print(B0.size())

            reg = self.fliter_reg(self.model)
            # print(reg)

            for j in range(self.S):
                loss_Bs = float(loss_Bs) + 0.1*self.loss(ListB[j], hr)
                loss_Rs = float(loss_Rs) + 0.1*self.loss(ListR[j], lr-hr)
            loss_B = self.loss(ListB[-1], hr)
            loss_R = 0.9 * self.loss(ListR[-1], lr-hr)
            loss_B0 = 0.1* self.loss(B0, hr)
            # loss = loss_B0 + loss_Bs  + loss_Rs + loss_B + loss_R
            loss = loss_B0 + loss_Bs  + loss_Rs + loss_B + loss_R + lamda * reg
            loss_Bs_all = loss_Bs_all + loss_Bs
            loss_B_all = loss_B_all + loss_B
            loss_Rs_all = loss_Rs_all + loss_Rs
            loss_R_all = loss_R_all + loss_R
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                ttt = 0
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{}\t{:.1f}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    n,
                    reg,
                    timer_model.release(),
                    timer_data.release()))

                # self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                #     (batch + 1) * self.args.batch_size,
                #     len(self.loader_train.dataset),
                #     self.loss.display_loss(batch),
                #     timer_model.release(),
                #     timer_data.release()))
            timer_data.tic()
        print(loss_Bs_all/cnt)
        print(loss_B_all / cnt)
        print(loss_Rs_all / cnt)
        print(loss_R_all / cnt)
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()

    def test(self):
        epoch = self.scheduler.last_epoch# + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    B0,ListB,ListR = self.model(lr, idx_scale)
                    sr = utility.quantize(ListB[-1], self.args.rgb_range)    # restored background at the last stage
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)
                        ##### for save rain_map
                        # rain_map = utility.quantize(ListR[-1], self.args.rgb_range)
                        # filename_map = filename + 'rain_map'   # for save rain_map
                        # self.ckp.save_results_map(filename_map, rain_map, scale)      # for save rain_map
                        ########################

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.model.train()
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        # device = torch.device('cpu' if self.args.cpu else 'cuda:3')
        device = torch.device('cpu' if self.args.cpu else 'cuda:'+ self.args.device)
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch #+ 1
            return epoch >= self.args.epochs
    
    def fliter_reg(self,model):  # calculate filter regularization
        reg = 0
        for layer in model.modules():
            if isinstance(layer, Fc.MyConv):
                reg += layer.rot_eq_loss
        return reg


    def rot(self,x,n):
        y = torch.rot90(x, n*2, [2, 3])
        return y

    def rotateAny(self, images, n):

        images_clone = images.clone()
        step = 360 / 8
        angle = n * step
        angle_rad = torch.deg2rad(torch.tensor(angle))
        # center = torch.tensor(images.shape[2:]) // 2

        theta = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                              [torch.sin(angle_rad), torch.cos(angle_rad), 0]])

        theta = theta.float()

        grid = F.affine_grid(theta.unsqueeze(0).expand(images_clone.size(0), -1, -1), images_clone.size(), align_corners=True).to(images.device)
        rotated_images = F.grid_sample(images_clone, grid, align_corners=True)

        return rotated_images

