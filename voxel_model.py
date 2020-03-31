import tensorflow as tf
import numpy as np
import math
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
#import helper
import tf_util
import tf_util_loss

def placeholder_inputs(batch_size, num_voxel):
	source_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_voxel, num_voxel, num_voxel))
	template_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_voxel, num_voxel, num_voxel))
	transformation_pl = tf.placeholder(tf.float32, shape=(batch_size, 4, 4))
	gt_transformation_pl = tf.placeholder(tf.float32, shape=(batch_size, 4, 4))
	return source_pointclouds_pl, template_pointclouds_pl, transformation_pl, gt_transformation_pl

def get_model(source_point_cloud, template_point_cloud, is_training, bn_decay=None):
	point_cloud = tf.concat([source_point_cloud, template_point_cloud],0)
	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value
	end_points = {}

	input_image = tf.expand_dims(point_cloud, -1)

	net = tf_util.conv3d(input_image, 32, [5, 5, 5],
						 padding='VALID', stride=[2, 2, 2],
						 bn=False, is_training=is_training,
						 scope='conv1', bn_decay=bn_decay)
	net = tf_util.conv3d(net, 32, [3, 3, 3],
						 padding='VALID', stride=[1, 1, 1],
						 bn=False, is_training=is_training,
						 scope='conv2', bn_decay=bn_decay)

	# Symmetric function: max pooling
	net = tf_util.max_pool3d(net, [2, 2, 2],
							 padding='VALID', scope='maxpool')
	net = tf.reshape(net, [batch_size, -1])
	print(net)
	source_global_feature = tf.slice(net, [0,0], [int(batch_size/2), 6912])
	template_global_feature = tf.slice(net, [int(batch_size/2),0], [int(batch_size/2), 6912])
	return source_global_feature, template_global_feature

def get_pose(source_global_feature, template_global_feature, is_training, bn_decay=None):
	net = tf.concat([source_global_feature,template_global_feature],1)
	net = tf_util.fully_connected(net, 1024, bn=False, is_training=is_training,scope='fc1', bn_decay=bn_decay)
	net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,scope='fc2', bn_decay=bn_decay)
	net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,scope='fc3', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope='dp4')
	predicted_transformation = tf_util.fully_connected(net, 7, activation_fn=None, scope='fc4')
	return predicted_transformation

def get_loss(predicted_transformation, batch_size, template_pointclouds_pl, source_pointclouds_pl):
	with tf.variable_scope('loss') as LossEvaluation:
		predicted_position = tf.slice(predicted_transformation,[0,0],[batch_size,3])
		predicted_quat = tf.slice(predicted_transformation,[0,3],[batch_size,4])

		# with tf.variable_scope('quat_normalization') as norm:
		norm_predicted_quat = tf.reduce_sum(tf.square(predicted_quat),1)
		norm_predicted_quat = tf.sqrt(norm_predicted_quat)
		norm_predicted_quat = tf.reshape(norm_predicted_quat,(batch_size,1))
		const = tf.constant(0.0000001,shape=(batch_size,1),dtype=tf.float32)
		norm_predicted_quat = tf.add(norm_predicted_quat,const)
		predicted_norm_quat = tf.divide(predicted_quat,norm_predicted_quat)

		transformed_predicted_point_cloud = helper.transformation_quat_tensor(source_pointclouds_pl, predicted_norm_quat,predicted_position)

		loss = tf_util_loss.chamfer(template_pointclouds_pl, transformed_predicted_point_cloud)
	return loss

def get_loss_v1(predicted_transformation, transformation_pl, gt_transformation_pl, batch_size):
	with tf.variable_scope('loss') as LossEvaluation:
		predicted_position = tf.slice(predicted_transformation,[0,0],[batch_size,3])
		predicted_quat = tf.slice(predicted_transformation,[0,3],[batch_size,4])

		# with tf.variable_scope('quat_normalization') as norm:
		norm_predicted_quat = tf.reduce_sum(tf.square(predicted_quat),1)
		norm_predicted_quat = tf.sqrt(norm_predicted_quat)
		norm_predicted_quat = tf.reshape(norm_predicted_quat,(batch_size,1))
		const = tf.constant(0.0000001,shape=(batch_size,1),dtype=tf.float32)
		norm_predicted_quat = tf.add(norm_predicted_quat,const)
		quat = tf.divide(predicted_quat,norm_predicted_quat)

		loss = tf.constant(0.0)
		transformed_data = tf.zeros([batch_size, 4, 4])		# Tensor to store transformed data.
		for i in range(quat.shape[0]):
			# Seperate each quaternion value.
			q0 = tf.slice(quat,[i,0],[1,1])
			q1 = tf.slice(quat,[i,1],[1,1])
			q2 = tf.slice(quat,[i,2],[1,1])
			q3 = tf.slice(quat,[i,3],[1,1])

			t0 = tf.slice(predicted_position, [i,0], [1,1])
			t1 = tf.slice(predicted_position, [i,1], [1,1])
			t2 = tf.slice(predicted_position, [i,2], [1,1])
			t3 = tf.constant(1.0,shape=[1,1])

			l0 = tf.constant(0.0,shape=[1,1])
			l1 = tf.constant(0.0,shape=[1,1])
			l2 = tf.constant(0.0,shape=[1,1])
			l3 = tf.constant(1.0,shape=[1,1])

			gt_T = tf.slice(gt_transformation_pl, [i,0,0], [1,4,4])
			gt_T = tf.reshape(gt_T, [4,4])
			prev_T = tf.slice(transformation_pl, [i,0,0], [1,4,4])
			prev_T = tf.reshape(prev_T, [4,4])

			# Convert quaternion to rotation matrix.
			# Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
				  # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
			T = [[q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2), t0],
			 	[2*(q1*q2+q0*q3), q0*q0+q2*q2-q1*q1-q3*q3, 2*(q2*q3-q0*q1), t1],
			 	[2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0*q0+q3*q3-q1*q1-q2*q2, t2],
			 	[l0, l1, l2, l3]]

			T = tf.reshape(T,[4,4]) 			# Convert T into a single tensor of shape 4x4.

			T = tf.matmul(T, prev_T)
			T = tf.linalg.inv(T)
			I = tf.eye(4)
			loss = tf.add(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.matmul(gt_T, T), I)))), loss)

		loss = tf.divide(loss, batch_size)
	return loss


if __name__=='__main__':
	with tf.Graph().as_default():
		inputs = tf.zeros((32,64,64,64))
		outputs = get_model(inputs, inputs, tf.constant(True))
		print(outputs)
