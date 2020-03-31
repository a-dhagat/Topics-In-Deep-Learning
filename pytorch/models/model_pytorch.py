import torch
import numpy as np

class VoxelFeatures(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = torch.nn.Conv3d(1, 32, kernel_size=(5,5,5), stride=(2,2,2))
		self.conv2 = torch.nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1))
		self.max_pool = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0))

	def forward(self, voxel):
		B, _, _, _, _ = voxel.shape
		net = self.conv1(voxel)
		net = self.conv2(net)
		net = self.max_pool(net)
		return net.view(B, -1)

class PoseEstimation(torch.nn.Module):
	def __init__(self, voxel_feature, Fin=2*32*6*6*6):
		super().__init__()
		layers = [Fin, 1024, 512, 256, 7]
		list_layers = []
		for idx in range(len(layers)-1):
			list_layers.append(torch.nn.Linear(layers[idx], layers[idx+1]))
		self.pose_est = torch.nn.Sequential(*list_layers)
		self.features = voxel_feature

	def voxelize(self, points, size=32):
		device = points.device
		# print(type(points))
		points_copy = points
		#points_copy = points_copy.requires_grad_()
		
		temp_def = torch.ones(points_copy.shape, dtype=torch.float64).cuda()
		points_copy = points_copy + temp_def*0.5
		occupany_grid = torch.zeros((points_copy.shape[0], size, size, size))
		#occupany_grid = occupany_grid.requires_grad_()
		#print(occupany_grid)
		B, N, _ = points.shape
		voxel_size = 1/(size*1.0)

		for idx in range(B):
			for pt_idx in range(N):
				x = int(points_copy[idx,pt_idx,0]/voxel_size)
				y = int(points_copy[idx,pt_idx,1]/voxel_size)
				z = int(points_copy[idx,pt_idx,2]/voxel_size)

				if x >= size: x = size-1
				if y >= size: y = size-1
				if z >= size: z = size-1
				if x < 0: x = 0
				if y < 0: y = 0
				if z < 0: z = 0

				occupany_grid[idx, x, y, z] = 1

		return occupany_grid.to(device)

	def voxelize_numpy(self, points, size=32):
		points_copy = (points.detach().cpu().numpy()).copy()
		points_copy = points_copy + 0.5
		occupany_grid = np.zeros((points_copy.shape[0], size, size, size))

		B, N, _ = points.shape

		voxel_size = 1/(size*1.0)

		for idx in range(B):
			for pt_idx in range(N):
				x = int(points_copy[idx,pt_idx,0]/voxel_size)
				y = int(points_copy[idx,pt_idx,1]/voxel_size)
				z = int(points_copy[idx,pt_idx,2]/voxel_size)

				if x >=size: x = size-1
				if y >=size: y = size-1
				if z >=size: z = size-1
				if x < 0: x = 0
				if y < 0: y = 0
				if x < 0: z = 0

				occupany_grid[idx, x, y, z] = 1
		occupany_grid = torch.tensor(occupany_grid, requires_grad=True, dtype=torch.float32)
		occupany_grid = occupany_grid.unsqueeze(-1)
		occupany_grid = occupany_grid.permute(0,4,1,2,3).cuda()
		return occupany_grid

	@staticmethod
	def pose2matrix(del_pose):
		# Convert Quaternion to Transformation matrix.
		matrix = torch.zeros(del_pose.shape[0],4,4)
		for i in range(del_pose.shape[0]):
			q0, q1, q2, q3 = del_pose[i,3], del_pose[i,4], del_pose[i,5], del_pose[i,6]
			matrix[i,0,3], matrix[i,1,3], matrix[i,2,3] = del_pose[i,0], del_pose[i,1], del_pose[i,2]
			R = [[q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
			 	 [2*(q1*q2+q0*q3), q0*q0+q2*q2-q1*q1-q3*q3, 2*(q2*q3-q0*q1)],
			 	 [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0*q0+q3*q3-q1*q1-q2*q2]]
			matrix[i, 0:3, 0:3] = torch.Tensor(R).view(3,3)
		return matrix

	def forward(self, source, template, maxItr=8):
		training = self.features.training
		template_voxel = self.voxelize_numpy(template)

		if training:
			source_voxel = self.voxelize_numpy(source)
			template_feat = self.features(template_voxel)
			source_feat = self.features(source_voxel)
		self.features.eval()

		template_feat = self.features(template_voxel)
		prediction = torch.eye(4).view(1,4,4).repeat(source.shape[0], 1, 1)

		for itr in range(maxItr):
			source_voxel = self.voxelize_numpy(source)

			source_feat = self.features(source_voxel)
			feat = torch.cat((source_feat, template_feat), 1)
			del_pose = self.pose_est(feat)

			Tmat = self.pose2matrix(del_pose)
			source = torch.bmm(Tmat[:,0:3,0:3].double().cuda(), source.permute(0,2,1).cuda())
			
			source = source.permute(0,2,1)
			prediction = torch.bmm(Tmat, prediction)

		source_voxel = self.voxelize_numpy(source)
		self.features.train(training)
		loss = source_voxel - template_voxel
		return (loss*loss).mean()