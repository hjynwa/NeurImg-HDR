import numpy as np
import os
import time
import torch
from torchvision.utils import make_grid
import cv2
from . import util
import wandb


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
            wandb.log({k + "-loss":v})
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # Save training samples to disk
    def save_image_to_disk(self, visuals, iteration, epoch):
        for label, image in visuals.items(): # ['input_ldr_rgb', 'input_im_up', 'gt_hdr_rgb', 'output_hdr_rgb']
            if('hdr' in label):
                image_numpy = util.tensor2hdr(image)
                tonemapped = util.hdr2tonemapped(image_numpy)
                # hdr_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s.exr' % (epoch, iteration, label))
                tonemap_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s_tmp.jpg' % (epoch, iteration, label))
                # cv2.imwrite(hdr_path, image_numpy[:,:,::-1])
                # util.writeEXR(image_numpy, hdr_path)
                cv2.imwrite(tonemap_path, tonemapped[:,:,::-1])
            elif('ldr' in label):
                image_numpy = image.detach().cpu().float().numpy()
                image_numpy = np.transpose(image_numpy[0], (1, 2, 0))
                image_numpy = (image_numpy + 1.0) / 2.0
                image_numpy = (image_numpy**(1/2.2)*255).astype(np.uint8)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s.jpg' % (epoch, iteration, label))
                cv2.imwrite(img_path, image_numpy[:,:,::-1])
            elif('im' in label):
                image_numpy = image.detach().cpu().float().numpy()
                spikes_img = np.transpose(image_numpy[0], (1, 2, 0))
                spikes_img = (spikes_img + 1.0) / 2.0
                spikes_img = (spikes_img-spikes_img.min()) / (spikes_img.max()-spikes_img.min())
                spikes_img = (spikes_img*255).astype(np.uint8)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s.jpg' % (epoch, iteration, label))
                cv2.imwrite(img_path, spikes_img)


    def newvideo_save_image(self, visuals, iteration, epoch):
        dest_dir = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d' % (epoch, iteration))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for label, image in visuals.items():
            if('hdr' in label):
                for i in range(len(image)):
                    image_numpy = util.tensor2hdr(image[i])
                    tonemapped = util.hdr2tonemapped(image_numpy)
                    tonemap_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d' % (epoch, iteration), '%04d_%s_tmp.jpg' % (i, label))
                    cv2.imwrite(tonemap_path, tonemapped[:,:,::-1])
            elif('ldr' in label):
                for i in range(len(image)):
                    image_numpy = image[i].detach().cpu().float().numpy()
                    image_numpy = np.transpose(image_numpy[0], (1, 2, 0))
                    image_numpy = (image_numpy + 1.0) / 2.0
                    image_numpy = (image_numpy**(1/2.2)*255).astype(np.uint8)
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d' % (epoch, iteration), '%04d_%s.jpg' % (i, label))
                    cv2.imwrite(img_path, image_numpy[:,:,::-1])
            elif('im' in label):
                for i in range(len(image)):
                    image_numpy = image[i].cpu().float().numpy()
                    spikes_img = np.transpose(image_numpy[0], (1, 2, 0))
                    spikes_img = (spikes_img + 1.0) / 2.0
                    spikes_img = (spikes_img-spikes_img.min()) / (spikes_img.max()-spikes_img.min())
                    spikes_img = (spikes_img*255).astype(np.uint8)
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d' % (epoch, iteration), '%04d_%s.jpg' % (i, label))
                    cv2.imwrite(img_path, spikes_img)
