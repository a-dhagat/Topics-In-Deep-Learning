import argparse
import numpy as np
import os
import sys
import helper
import torch
from models import model_pytorch as network

parser = argparse.ArgumentParser()
parser.add_argument('-mode','--mode', required=False, type=str, default='no_mode', help='mode: train or test')
parser.add_argument('-log','--log_dir', default='log_train', help='Log dir [default: log]')
parser.add_argument('--model_path', type=str, default='log_multi_catg_noise/model300.ckpt', help='Path of the weights (.ckpt file) to be used for test')

# Dataset Settings.
parser.add_argument('-noise','--add_noise', type=bool, default=False, help='Use of Noise in source data in training')
parser.add_argument('--use_partial_data', type=bool, default=False, help='Use of Partial Data for Registration')
parser.add_argument('--use_pretrained_model', type=bool, default=False, help='Use a pretrained model of airplane to initialize the training.')
parser.add_argument('--use_random_poses', type=bool, default=False, help='Use of random poses to train the model in each batch')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--num_voxel', type=int, default=32, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--centroid_sub', type=bool, default=True, help='Centroid Subtraction from Source and Template before Pose Prediction.')

# Training Settings.
parser.add_argument('--max_epoch', type=int, default=301, help='Epoch to run [default: 250]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--device', type=str, default='cuda')

parser.add_argument('--data_dict', type=str, default='train_data',help='Templates data used for training network')
parser.add_argument('--train_poses', type=str, default='itr_net_train_data45.csv', help='Poses for training')
parser.add_argument('--eval_poses', type=str, default='itr_net_eval_data45.csv', help='Poses for evaluation')
args = parser.parse_args()

# Model Import
LOG_DIR = args.log_dir

# Take backup of all files used to train the network with all the parameters.
if args.mode == 'train':
	if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)			# Create Log_dir to store the log.
	LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')# Create a text file to store the loss function data.
	LOG_FOUT.write(str(args)+'\n')

# Write all the data of loss function during training.
def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)

class Action:
	def __init__(self):
		self.maxItr = 8

	def create_model(self):
		features = network.VoxelFeatures()
		return network.PoseEstimation(features)

	def train_one_epoch(self, model, poses, templates, optimizer, BATCH_SIZE=32):
		num_batches = int(templates.shape[0]/BATCH_SIZE)
		total_loss = 0.0
		weight = 4.0
		for batch_idx in range(num_batches):
			start_idx = batch_idx*BATCH_SIZE
			end_idx = (batch_idx+1)*BATCH_SIZE
			
			template_data = templates[start_idx:end_idx]
			source_data = helper.apply_transformation(template_data, poses[start_idx:end_idx])

			template_data = template_data - np.mean(template_data, axis=1, keepdims=True)
			source_data = source_data - np.mean(source_data, axis=1, keepdims=True)

			source_data = torch.from_numpy(source_data).cuda().double()
			template_data = torch.from_numpy(template_data).cuda().double()

			loss = model(source_data, template_data, self.maxItr) * weight
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		return total_loss/num_batches

	def eval_one_epoch(self, model, poses, templates, BATCH_SIZE=32):
		num_batches = int(poses.shape[0]/BATCH_SIZE)
		total_loss = 0.0
		weight = 4
		with torch.no_grad():
			for batch_idx in range(num_batches):
				start_idx = batch_idx*BATCH_SIZE
				end_idx = (batch_idx+1)*BATCH_SIZE
				
				template_data = templates[start_idx:end_idx]
				source_data = helper.apply_transformation(template_data, poses[0:BATCH_SIZE])

				template_data = template_data - np.mean(template_data, axis=1, keepdims=True)
				source_data = source_data - np.mean(source_data, axis=1, keepdims=True)

				source_data = torch.from_numpy(source_data).cuda().double()
				template_data = torch.from_numpy(template_data).cuda().double()

				loss = model(source_data, template_data, self.maxItr) * weight

				total_loss += loss.item()
		return total_loss/num_batches

def save_checkpoint(state, filename, suffix):
	torch.save(state, '{}_{}.pth'.format(filename, suffix))

def train():
	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	action = Action()
	model = action.create_model()
	model.to(args.device)
	model.cuda()

	min_loss = float('inf')
	learnable_params = filter(lambda p: p.requires_grad, model.parameters())
	if args.optimizer == 'adam': optimizer = torch.optim.Adam(learnable_params)
	else: optimizer = torch.optim.SGD(learnable_params, lr=0.1)

	templates = helper.loadData(args.data_dict)
	poses = helper.read_poses(args.data_dict, args.train_poses)
	eval_poses = helper.read_poses(args.data_dict, args.eval_poses)
	print(templates.shape, poses.shape, eval_poses.shape)

	for epoch in range(args.max_epoch):
		log_string("############## EPOCH: %0.4d ##############"%epoch)
		train_loss = action.train_one_epoch(model, poses, templates, optimizer)
		eval_loss = action.eval_one_epoch(model, eval_poses, templates)

		log_string("Training Loss: {} and Evaluation Loss: {}".format(train_loss, eval_loss))

		is_best = eval_loss<min_loss

		snap = {'epoch': epoch + 1,
				'model': model.state_dict(),
				'min_loss': min_loss,
				'optimizer' : optimizer.state_dict()}
		
		if is_best:
			save_checkpoint(snap, os.path.join(LOG_DIR, 'model'), 'snap_best')
			save_checkpoint(model.state_dict(), os.path.join(LOG_DIR, 'model'), 'model_best')
			log_string("Best Evaluation Loss: {}".format(eval_loss))

		save_checkpoint(snap, os.path.join(LOG_DIR, 'model'), 'snap_last')
		save_checkpoint(model.state_dict(), os.path.join(LOG_DIR, 'model'), 'model_last')	 

if __name__ == "__main__":
	if args.mode == 'train':
		train()
	else:
		print("Specify the mode")