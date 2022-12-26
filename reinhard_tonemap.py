import os
import argparse
import cv2
import numpy as np 
import Imath, OpenEXR
from skimage import color 
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', default='', type=str)
    parser.add_argument('--src', default='exr_results', type=str)
    parser.add_argument('--tm_gamma', default=0.85, type=float)
    parser.add_argument('--sat_fac', default=1.0, type=float) # saturation factor
    parser.add_argument('--key_fac', default=0.1, type=float) 
    parser.add_argument('--tm_img', default=False, action='store_true')
    args = parser.parse_args()
    
    return args


class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def readEXR(hdrfile):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    hdr_t = OpenEXR.InputFile(hdrfile)
    dw = hdr_t.header()['dataWindow']
    size = (dw.max.x-dw.min.x+1, dw.max.y-dw.min.y+1)
    rstr = hdr_t.channel('R', pt)
    gstr = hdr_t.channel('G', pt)
    bstr = hdr_t.channel('B', pt)
    r = np.frombuffer(rstr, dtype=np.float32)
    r.shape = (size[1], size[0])
    g = np.frombuffer(gstr, dtype=np.float32)
    g.shape = (size[1], size[0])
    b = np.frombuffer(bstr, dtype=np.float32)
    b.shape = (size[1], size[0])
    res = np.stack([r,g,b], axis=-1)
    imhdr = np.asarray(res)
    return imhdr

def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        R = (img[:,:,0]).astype(np.float16).tostring()
        G = (img[:,:,1]).astype(np.float16).tostring()
        B = (img[:,:,2]).astype(np.float16).tostring()
        out.writePixels({'R' : R, 'G' : G, 'B' : B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)

def crop(img):
    h, w = img.shape[:2]
    start = int((w-h)/2)
    cropped = img[:,start:start+h, :]
    cropped = cv2.resize(cropped, (512, 512), cv2.INTER_LINEAR)

    return cropped

def whiteBalance(img):
    h, w = img.shape[:2]
    img = np.transpose(img, (2,0,1))
    img = np.reshape(img, (3, -1))

    r_max = img[0].max()
    g_max = img[1].max()
    b_max = img[2].max()
    
    mat = [[g_max/r_max, 0, 0], [0, 1.0, 0], [0,0,g_max/b_max]]
    img_wb = np.dot(mat, img)
    img_wb = np.reshape(img_wb, (3, h, w))
    img_wb = np.transpose(img_wb, (1,2,0))
    
    return img_wb, mat


def whiteBalance_mat(img, mat):
    h, w = img.shape[:2]
    img = np.transpose(img, (2,0,1))
    img = np.reshape(img, (3, -1))

    img_wb = np.dot(mat, img)
    img_wb = np.reshape(img_wb, (3, h, w))
    img_wb = np.transpose(img_wb, (1,2,0))
    
    return img_wb

if __name__ == "__main__":
    args = parse_args()
    cfgs = vars(args)
    base_dir = './results'   
    epsilon = 1e-6
    # if_img = cfgs['tm_img']
    # key_fac = cfgs.get('key_fac', 0.100)
    # tm_gamma = cfgs.get('tm_gamma', 0.8500)
    # src_dir_name = cfgs['src']
    # sat_fac = cfgs.get('sat_fac', 1.0)
    dir_name = cfgs.get('in_dir')
    
    if_img = False
    key_fac = 0.5
    tm_gamma = 1.4
    src_dir_name = 'exr_results'
    sat_fac = 1.0
    
    src_path = os.path.join(base_dir, dir_name, src_dir_name)
    tmp_dir = os.path.join(base_dir, dir_name, '%s_Reinhard_PyTM' % src_dir_name)
    # exr_dir = os.path.join(base_dir, 'exr')
    
    files = sorted(os.listdir(src_path))

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for i, file in enumerate(tqdm(files[:], ascii=True)):
        if file.endswith('.jpg'):
            continue
        if file.endswith('.hdr'):
            hdr_img = cv2.imread(os.path.join(src_path, file), -1)
        elif file.endswith('.exr'):
            hdr_img = readEXR(os.path.join(src_path, file))[:,:,::-1]
        
        if hdr_img.min() < 0:
            hdr_img = hdr_img - hdr_img.min()
        # hdr_img = (hdr_img-hdr_img.min())/(hdr_img.max()-hdr_img.min())
        
        img_lab = color.rgb2lab(hdr_img).astype(np.float32)

        if i == 0 or if_img: # not video, no temporal smoothing
            log_sum_prev = np.log(epsilon + img_lab[:,:,0].reshape(-1)).sum()
            key = key_fac
        else:
            h, w, _ = img_lab.shape
            log_sum_cur = np.log(img_lab[:,:,0].reshape(-1) + epsilon).sum()
            log_avg_cur = np.exp(log_sum_cur / (h * w))
            log_avg_temp = np.exp((log_sum_cur + log_sum_prev) / (2.0 * h * w))
            key = key_fac * log_avg_cur / log_avg_temp
            log_sum_prev = log_sum_cur
        if file.endswith('.hdr'):
            tmp_path = os.path.join(tmp_dir, file.replace('.hdr', '.jpg'))
        elif file.endswith('.exr'):
            tmp_path = os.path.join(tmp_dir, file.replace('.exr', '.jpg'))
        command = 'luminance-hdr-cli -l %s --tmo reinhard02 --tmoR02Key %f -G %f -S %f -o %s > /dev/null' % (os.path.join(src_path, file), key, tm_gamma, sat_fac, tmp_path)
        os.system(command)
        