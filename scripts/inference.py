import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp

from torch.optim import Adam

from criteria import id_loss, w_norm, moco_loss

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def run():
	test_opts = TestOptions().parse()

	if test_opts.resize_factors is not None:
		assert len(
			test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
										'downsampling_{}'.format(test_opts.resize_factors))
		out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
										'downsampling_{}'.format(test_opts.resize_factors))
	else:
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
		out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

	os.makedirs(out_path_results, exist_ok=True)
	os.makedirs(out_path_coupled, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	if 'output_size' not in opts:
		opts['output_size'] = 1024
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()

	# random input code
	np.random.seed(seed=2022)
	# incodes = np.random.randn(1, 512).astype('float32')
	incodes = np.random.randn(1, 18, 512).astype('float32')
	optimized_codes = torch.tensor(incodes).float()
	optimized_codes = torch.nn.Parameter(optimized_codes.data)	
	optimizer = Adam([optimized_codes], lr=0.05)

	# sample image
	real_img = Image.open(opts.data_path)
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	transform_inference = transforms_dict['transform_inference']
	real_img = transform_inference(real_img).cuda()

	max_iter = 10000
	L1_loss = torch.nn.L1Loss()
	# ID_loss = id_loss.IDLoss()

	# train
	for iter in range(max_iter):
		y_hat, _ = net.decoder([optimized_codes],
								input_is_latent=True,
								randomize_noise=False)
		y_hat = net.face_pool(y_hat)
		y_hat.cuda()
		result = tensor2im(y_hat[0])
		z_loss = L1_loss(y_hat[0], real_img)
		# + ID_loss(result, real_img, optimized_codes)
		
		optimizer.zero_grad()
		z_loss.backward()
		optimizer.step()
		
		if iter % 10 == 0:
			print(z_loss.item())
			if iter % 100 == 0:
				result.save('./results/wplus3/result_{}.png'.format(iter))
		
		with open('./results/wplus3/loss.txt', 'a') as f:
			text = f"{z_loss}\n"
			f.write(text)
	result.save('./results/wplus3/result_{}.png'.format(max_iter))

if __name__ == '__main__':
	run()
