"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config
from trainer import UNIT_Trainer
import matplotlib.pyplot as plt
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/unit_noise2clear-bn.yaml', help="net configuration")
parser.add_argument('--input', type=str, default = '/home/user/data4/chongxin/LIR-for-Unsupervised-IR/dataset/motion3/Celeba_A/', help="input image path")

parser.add_argument('--output_folder', type=str, default='/home/user/data4/chongxin/LIR-for-Unsupervised-IR/output3/outputs/test_all/motion', help="output image path")
parser.add_argument('--checkpoint', type=str, default='/home/user/data4/chongxin/LIR-for-Unsupervised-IR/output/outputs/unit_noise2clear-bn/checkpoints/gen_00300000.pt',
                    help="checkpoint of autoencoders") 
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
parser.add_argument('--psnr', action="store_false", help='is used to compare psnr')
parser.add_argument('--ref', type=str, default='J:\\Public_DataSet\\Kodak\\original\\kodim04.png', help='cmpared refferd image')
opts = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loaderopts.trainer == 'UNIT':
trainer = UNIT_Trainer(config)
state_dict = torch.load(opts.checkpoint, map_location='cpu')
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode_cont  # encode function
decode = trainer.gen_b.dec_cont  # decode function

clear_path = '/home/user/data4/chongxin/LIR-for-Unsupervised-IR/dataset/motion3/Celeba_B/'
clearimglist = sorted(os.listdir(clear_path))

if not os.path.exists(opts.input):
    raise Exception('input path is not exists!')
imglist = sorted(os.listdir(opts.input))

transform_list = [transforms.ToTensor()]  #
transform_list = [transforms.RandomCrop((240, 240))] + transform_list
transform = transforms.Compose(transform_list)

for i, file in enumerate(imglist):
    if not file.endswith('.png'):
        continue
    print(file)
    filepath = opts.input + '/' + file

    clearfilepath = clear_path + clearimglist[i]
    clearimage = transform(Image.open(clearfilepath)).unsqueeze(0).cuda()

    image = transform(Image.open(filepath)).unsqueeze(0).cuda()  #torch.Size([1, 1, 256, 320])
        # Start testing
    h,w = image.size(2),image.size(3)
    if h > 800 or w > 800:
        continue
    pad_h = h % 4
    pad_w = w % 4
    image = image[:,:,0:h-pad_h, 0:w - pad_w]  #torch.Size([1, 1, 256, 320])


    h_a = trainer.gen_a.encode_cont(image)
    h_a_sty = trainer.gen_a.encode_sty(image)
    h_b = trainer.gen_b.encode_cont(clearimage)

    h_ba_cont = torch.cat((h_b, h_a_sty), 1)
    h_aa_cont = torch.cat((h_a, h_a_sty), 1)

    x_ba = trainer.gen_a.decode_recs(h_ba_cont)
    x_b_recon = trainer.gen_b.decode_cont(h_b)

    x_ab = trainer.gen_b.decode_cont(h_a)
    x_a_recon = trainer.gen_a.decode_recs(h_aa_cont)

    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)
    path = os.path.join(opts.output_folder, file)
    # final = torch.cat((image,outputs,x_a_recon), dim=3) # torch.Size([1, 1, 512, 320])

    final1 = torch.cat((image,x_ab,x_a_recon), dim=3) # torch.Size([1, 1, 512, 320])
    final2 = torch.cat((clearimage,x_ba,x_b_recon), dim=3) # torch.Size([1, 1, 512, 320])

    vutils.save_image(final1.data, path, padding=0, normalize=True)

    path_clear = os.path.join('/home/user/data4/chongxin/LIR-for-Unsupervised-IR/output3/outputs/test_all/clear',clearimglist[i])
    vutils.save_image(final2.data, path_clear, padding=0, normalize=True)


    # if opts.psnr:
    #     outputs = torch.squeeze(outputs_back)
    #     outputs = outputs.permute(1, 2, 0).to('cpu', torch.float32).numpy()
    #     # outputs = outputs.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #     ref = Image.open(opts.ref).convert('RGB')
    #     ref = np.array(ref) / 255.
    #     noi = Image.open(opts.input).convert('RGB')
    #     noi = np.array(noi) / 255.
    #     rmpad_h = noi.shape[0] % 4
    #     rmpad_w = noi.shape[1] % 4

    #     pad_h = ref.shape[0] % 4
    #     pad_w = ref.shape[1] % 4

        # if rmpad_h != 0 or pad_h != 0:
        #     noi = noi[0:noi.shape[0]-rmpad_h,:,:]
        #     ref = ref[0:ref.shape[0]-pad_h,:,:]
        # if rmpad_w != 0 or pad_w != 0:
        #     noi = noi[:, 0:noi.shape[1]-rmpad_w,:]
        #     ref = ref[:, 0:ref.shape[1]-pad_w,:]
            
        # psnr = compare_psnr(ref, outputs)
        # ssim = compare_ssim(ref, outputs, multichannel=True)
        # print('psnr:{}, ssim:{}'.format(psnr, ssim))
        # plt.figure('ref')
        # plt.imshow(ref, interpolation='nearest')
        # plt.figure('out')
        # plt.imshow(outputs, interpolation='nearest')
        # plt.figure('in')
        # plt.imshow(noi, interpolation='nearest')
        # plt.show()

    
    

