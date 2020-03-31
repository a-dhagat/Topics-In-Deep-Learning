# Topics-In-Deep-Learning  
This project explores how 2 point clouds can be registered (aligned) by taking into account their spatial properties.  
The point cloud is first converted to an occupancy grid (binary grid - 0 if a voxel is empty and 1 if it is filled). Then 3D convolutions are applied to it such that we can extract a feature vector from it.  
Both point clouds share convolution weights, and are then concatenated, like in a Siamese architecture, and then through _fc_ layers, regressed to a pose - (rotation and translation).  

This, along with a discussion of loss functions, preserving continuity during the binary discretization step, and results, is discussed in detail in the paper.
