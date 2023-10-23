import torch
import numpy as np
import os
from collections import OrderedDict
from .base_model import BaseModel
from . import networks
from .vgg import Vgg16
from util.util import tensor_tonemap

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                
    return (batch - mean) / std


def Gram_matrix(input):
    a,b,c,d = input.size()
    features = input.reshape(a*b, c*d)
    G = torch.mm(features, features.t())

    return G.div(a*b*c*d)


def load_module_dict(pth_path, gpu_ids):
    kwargs={'map_location': lambda storage, loc: storage.cuda(gpu_ids)}
    state_dict = torch.load(pth_path)

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k # remove `module.`
        new_state_dict[name] = v
    # load params
    return new_state_dict


class ImageModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance')
        if is_train:
            parser.add_argument('--lambda_L1_color', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_perc_color', type=float, default=5.0, help='weight for perceptual loss')
            parser.add_argument('--lambda_GAN', type=float, default=3.0, help='weight for generator loss in GAN')
            parser.add_argument('--lambda_D', type=float, default=10.0, help='weight for discriminator loss in GAN')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.netUpsample = networks.define_UpsampleNet(gpu_ids=self.gpu_ids, scale=opt.up_scale)
        self.netLumiFusion = networks.define_G(init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.netColor = networks.define_ColorNet(netColor=opt.netColor, norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, n_blocks=opt.colornet_n_blocks, state_nc=opt.state_nc, gpu_ids=self.gpu_ids)
        
        if self.opt.loss_type == 'l1+perc+gan':
            self.loss_names = ['G_L1_color', 'G_perc_color', 'G_GAN', 'D']

        if opt.phase == "infer": # for infer
            self.visual_names = ['input_ldr_rgb', 'input_im', 'output_hdr_rgb']
            self.model_names = ['LumiFusion', 'Color']
            self.netLumiFusion = self.load_pretrained_networks(netType='LumiFusion')
            self.netColor = self.load_pretrained_networks(netType='Color')
        else: # for training
            self.visual_names = ['input_ldr_rgb', 'input_im', 'gt_hdr_rgb', 'output_hdr_rgb']
            self.model_names = ['Color', 'D']
            self.netLumiFusion = self.load_pretrained_networks(netType='LumiFusion', isTrain=True)
            self.netD = networks.define_D(gpu_ids=self.gpu_ids)
            
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.vgg = Vgg16(requires_grad=False).to(opt.gpu_ids[0])
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            
            self.optimizer_ColorNet = torch.optim.Adam(self.netColor.parameters(), lr=opt.lr_colornet, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_ColorNet)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_colornet, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        self.input_ldr_y = input['input_ldr_y'].to(self.device)
        self.input_ldr_rgb = input['input_ldr_rgb'].to(self.device)
        self.input_im = input['input_im'].to(self.device)
        if (self.opt.phase != "infer"):
            self.gt_hdr_y = input['gt_hdr_y'].to(self.device)
            self.gt_hdr_uv = input['gt_hdr_uv'].to(self.device)
            self.gt_hdr_rgb = input['gt_hdr_rgb'].to(self.device)
        self.input_ldr_u = input['input_ldr_u'].to(self.device)
        self.input_ldr_v = input['input_ldr_v'].to(self.device)
        self.image_paths = input['paths']
        

    def forward(self):
        self.input_im_up = self.netUpsample(self.input_im)
        self.output_hdr_y, self.att_map = self.netLumiFusion(self.input_ldr_y, self.input_im_up.detach())
        self.output_hdr_rgb = self.netColor(self.output_hdr_y.detach(), self.input_ldr_u, self.input_ldr_v)
        
    
    def load_pretrained_networks(self, netType, isTrain=False):
        if netType == 'LumiFusion':
            net = self.netLumiFusion
            if not isTrain:
                load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'luminance_fusion_net.pth')
            else:
                load_path = os.path.join(self.opt.pretrained_lfn, 'luminance_fusion_net.pth')
        elif netType == 'Color':
            net = self.netColor
            load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'chrominance_compensation_net.pth')
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)
        net.eval()
        return net


    def backward_D(self):
        self.tmp_gt_hdr_rgb = tensor_tonemap(self.gt_hdr_rgb)
        self.tmp_output_hdr_rgb = tensor_tonemap(self.output_hdr_rgb)
        
        pred_fake = self.netD(self.tmp_output_hdr_rgb.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_D
        # Real
        # real_AB = torch.cat((self.fake_cat_B, self.real_color_B), 1)
        pred_real = self.netD(self.tmp_gt_hdr_rgb.detach())
        self.loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_D
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        

    def backward_G(self):
        # Compute L1 loss
        self.tmp_gt_hdr_rgb = tensor_tonemap(self.gt_hdr_rgb)
        self.tmp_output_hdr_rgb = tensor_tonemap(self.output_hdr_rgb)
        self.loss_G_L1_color = self.criterionL1(self.tmp_output_hdr_rgb, self.tmp_gt_hdr_rgb) * self.opt.lambda_L1_color

        # Compute perceptual loss on colored hdr
        output_hdr_features_color = self.vgg(normalize_batch(self.tmp_output_hdr_rgb))
        gt_hdr_features_color = self.vgg(normalize_batch(self.tmp_gt_hdr_rgb))
        
        self.loss_G_perc_color = 0.0
        for f_x, f_y in zip(output_hdr_features_color, gt_hdr_features_color):
            self.loss_G_perc_color += torch.mean((f_x - f_y)**2)
            G_x = Gram_matrix(f_x)
            G_y = Gram_matrix(f_y)
            self.loss_G_perc_color += torch.mean((G_x - G_y)**2)
        self.loss_G_perc_color = self.loss_G_perc_color * self.opt.lambda_perc_color

        # combine loss and calculate gradients
        if self.opt.loss_type == 'l1+perc+gan':
            pred_fake = self.netD(self.output_hdr_rgb.detach())
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
            self.loss_G_ColorNet = self.loss_G_GAN + self.loss_G_L1_color + self.loss_G_perc_color
        
        # self.loss_G.backward()
        self.loss_G_ColorNet.backward()


    def optimize_parameters(self):
        self.forward()
        
        if('gan' in self.opt.loss_type):
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            self.set_requires_grad(self.netD, False)
        
        # update G
        self.optimizer_ColorNet.zero_grad()
        self.backward_G()
        self.optimizer_ColorNet.step()
