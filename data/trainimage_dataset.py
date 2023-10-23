import os.path
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from data.base_dataset import BaseDataset, get_pairwise_transform
from data.image_folder import make_dataset
from util.util import readEXR, writeEXR, whiteBalance


class TrainImageDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_ldr = os.path.join(opt.dataroot, 'LDR/' + opt.phase) 
        self.dir_hdr = os.path.join(opt.dataroot, 'HDR/' + opt.phase)
        self.dir_im = os.path.join(opt.dataroot, 'IM/' + opt.phase)

        self.ldr_paths = sorted(make_dataset(self.dir_ldr, opt.max_dataset_size))
        self.hdr_paths = sorted(make_dataset(self.dir_hdr, opt.max_dataset_size))
        self.im_paths = sorted(make_dataset(self.dir_im, opt.max_dataset_size))
        self.ldr_size = len(self.ldr_paths)

        self.transform = get_pairwise_transform(self.opt, convert=False)

    def __getitem__(self, index):
        idx = index % self.ldr_size
        ldr_path = self.ldr_paths[idx]
        hdr_path = self.hdr_paths[idx]
        im_path = self.im_paths[idx]
        
        # ----------- Load HDR Image -------------
        if(hdr_path[-4:] == '.hdr'):
            hdr_img = cv2.imread(hdr_path, -1)
            hdr_img = hdr_img[:,:,::-1]
        else:
            hdr_img = readEXR(hdr_path)
            if hdr_img.min() < 0:
                hrd_img = hdr_img-hdr_img.min()
        hdr_img = (hdr_img / hdr_img.max()).astype(np.float32)
        img_h, img_w =  hdr_img.shape[:2]
        
        # white balance
        # hdr_img = whiteBalance(hdr_img).astype(np.float32)
                
        # ----------- Load LDR Image -------------
        ldr_img = Image.open(ldr_path).convert('RGB')
        if(ldr_img.size != (img_w, img_h)):
            ldr_img = ldr_img.resize((img_w, img_h))
        ldr_img = np.array(ldr_img)
        ldr_img = ldr_img / 255.0
        ldr_img = (ldr_img)**2.2
        
        # ----------- Load Vidar Intensity Map -------------
        im = np.load(im_path).astype(np.float32)
        im = cv2.resize(im, (self.opt.resolution//2,self.opt.resolution//2), interpolation=cv2.INTER_LINEAR) # INTER_CUBIC is bad, especially for exr files!
        im = (im * 2.0 - 1.0).astype(np.float32)
        im_tensor = transforms.ToTensor()(im)
        
        # apply image transformation
        ldr_hdr_img = np.concatenate((ldr_img, hdr_img), axis=2)
        ldr_hdr_img = self.transform(ldr_hdr_img)

        ldr_img, hdr_img = np.split(ldr_hdr_img, 2, 2)
        oh, ow = hdr_img.shape[:2]
        low_x = int((oh-self.opt.resolution) / 2)
        low_y = int((ow-self.opt.resolution) / 2)
        ldr_crop = ldr_img[low_x:low_x+self.opt.resolution, low_y:low_y+self.opt.resolution, :]
        ldr_norm = ldr_crop * 2.0 - 1.0
        data_ldr_rgb = transforms.ToTensor()(ldr_norm.astype(np.float32))
        
        ###################### HDR To YUV #####################
        hdr_crop = hdr_img[low_x:low_x+self.opt.resolution, low_y:low_y+self.opt.resolution, :]
        hdr_yuv = cv2.cvtColor(hdr_crop.astype(np.float32), cv2.COLOR_RGB2YUV)
        hdr_y = hdr_yuv[:,:,0]
        hdr_uv = hdr_yuv[:,:,1:]
        
        data_hdr_y = transforms.ToTensor()(hdr_y.astype(np.float32))
        data_hdr_uv = transforms.ToTensor()(hdr_uv.astype(np.float32))
        data_hdr_rgb = transforms.ToTensor()(hdr_crop.astype(np.float32))
        data_hdr_yuv = transforms.ToTensor()(hdr_yuv.astype(np.float32))
        
        ###################### LDR To YUV #####################
        ldr_yuv = cv2.cvtColor(ldr_crop.astype(np.float32), cv2.COLOR_RGB2YUV)
        ldr_y = ldr_yuv[:,:,0] # [0.0, 1.0]
        ldr_u = ldr_yuv[:,:,1] # [-0.5, 0.5]
        ldr_v = ldr_yuv[:,:,2] # [-0.5, 0.5]

        ldr_y = ldr_y * 2.0 - 1.0
        data_ldr_y = transforms.ToTensor()(ldr_y.astype(np.float32))
        data_ldr_u = transforms.ToTensor()(ldr_u.astype(np.float32))
        data_ldr_v = transforms.ToTensor()(ldr_v.astype(np.float32))
        
        return {'input_ldr_y': data_ldr_y, 'gt_hdr_y': data_hdr_y, 'input_im': im_tensor, 
                'paths': ldr_path,
                'input_ldr_u': data_ldr_u, 'input_ldr_v': data_ldr_v, 'gt_hdr_uv': data_hdr_uv, 'gt_hdr_yuv': data_hdr_yuv,
                'input_ldr_rgb': data_ldr_rgb,
                'gt_hdr_rgb': data_hdr_rgb
                }
        

    def __len__(self):
        """Return the total number of images."""
        return self.ldr_size
