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


class LFNModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance')
        if is_train:
            parser.add_argument('--lambda_L1_Y', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_perc_Y', type=float, default=5.0, help='weight for perceptual loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.netUpsample = networks.define_UpsampleNet(gpu_ids=self.gpu_ids, scale=opt.up_scale)
        self.netLumiFusion = networks.define_G(init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.opt.loss_type == 'l1+perc':
            self.loss_names = ['G_L1_Y', 'G_perc_Y']

        self.visual_names = ['input_ldr_y', 'input_im', 'gt_hdr_y', 'output_hdr_y']
        self.model_names = ['LumiFusion']
        
        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()
        self.vgg = Vgg16(requires_grad=False).to(opt.gpu_ids[0])
        self.optimizer_LumiFusion = torch.optim.Adam(self.netLumiFusion.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_LumiFusion)

    def set_input(self, input):
        self.input_ldr_y = input['input_ldr_y'].to(self.device)
        self.input_im = input['input_im'].to(self.device)
        self.gt_hdr_y = input['gt_hdr_y'].to(self.device)
        self.image_paths = input['paths']
        

    def forward(self):
        self.input_im_up = self.netUpsample(self.input_im)
        self.output_hdr_y, self.att_map = self.netLumiFusion(self.input_ldr_y, self.input_im_up.detach())


    def backward_G(self):
        # Compute L1 loss
        self.tmp_gt_hdr_y = tensor_tonemap(self.gt_hdr_y)
        self.tmp_output_hdr_y = tensor_tonemap(self.output_hdr_y)
        self.loss_G_L1_Y = self.criterionL1(self.tmp_output_hdr_y, self.tmp_gt_hdr_y) * self.opt.lambda_L1_Y

        # Compute perceptual loss on colored hdr
        output_hdr_features_Y = self.vgg(normalize_batch(self.tmp_output_hdr_y))
        gt_hdr_features_Y = self.vgg(normalize_batch(self.tmp_gt_hdr_y))
        
        self.loss_G_perc_Y = 0.0
        for f_x, f_y in zip(output_hdr_features_Y, gt_hdr_features_Y):
            self.loss_G_perc_Y += torch.mean((f_x - f_y)**2)
            G_x = Gram_matrix(f_x)
            G_y = Gram_matrix(f_y)
            self.loss_G_perc_Y += torch.mean((G_x - G_y)**2)
        self.loss_G_perc_Y = self.loss_G_perc_Y * self.opt.lambda_perc_Y
        
        self.loss_G_LumiFusionNet = self.loss_G_L1_Y + self.loss_G_perc_Y
        self.loss_G_LumiFusionNet.backward()


    def optimize_parameters(self):
        self.forward()
        
        # update G
        self.optimizer_LumiFusion.zero_grad()
        self.backward_G()
        self.optimizer_LumiFusion.step()
