import os
import numpy as np
import time
import pickle
import math
from queue import SimpleQueue

import torch


class Octree_loss():
    """Octree is the data structure established for sampling points based on an optional feature (e.g. variance) in a random data crop.
    """

    def __init__(self, data_dir="./data", data_filename="rb2d_ra1e6_s42.npz", data=None, loss=None, loss_std=None, crop_size=(16,128,128), n_min_vol_per_crop=1):
        """
        Initialize Octree
        Args:
          data_dir: str, path to the dataset folder, default="./data"
          data_filename: str, name of the dataset file, default="rb2d_ra1e6_s42.npz"
          data: array, [c, t, z, x]
          mean: float, data mean
          std: float, data standard deviation
          crop_size: tuple, size of the data crop, default=(16,128,128)
          n_min_vol_per_crop: int, number of minimum volumes in a data crop
          loss: tensor, [c,t,z,x], l1_loss distribution
          loss_std: tensor, [c] , l1_loss distribution std
        """

        self.data_dir = data_dir
        self.data_filename = data_filename
        self.data = data # [c, t, z, x]
        self.loss = loss
        self.loss_std = loss_std
        self.crop_size = crop_size
        self.n_min_vol_per_crop = n_min_vol_per_crop

        self.depth = 0 # depth of recursion
        self.depth_now = 0 
        
        self.root = None # octree
        self.leaves = [] # all the leaf nodes


        octree_name = os.path.join(self.data_dir, self.data_filename+'.octree_loss.pkl')
        octree_leaves_name = os.path.join(self.data_dir, self.data_filename+'.leaves_loss.pkl')

        if not os.path.exists(octree_name):
            print('build octree for %s ...' %(data_filename))
            self.build_octree() # build octree if do not find one
        else:
            print('load octree of %s ...' %(data_filename))
            with open(octree_name, 'rb') as f:
                self.root = pickle.loads(f.read()) # load octree if find one
            with open(octree_leaves_name, 'rb') as f:
                self.leaves = pickle.loads(f.read()) # load all the leaf nodes

    def build_octree(self):
        """Build octree corresponding to the whole volumetric data.
        """

        time_start = time.time()

        # initialize octree
        if self.data is None:
            npdata = np.load(os.path.join(self.data_dir, self.data_filename)) # load data from file
            self.data = np.stack([npdata['p'], npdata['b'], npdata['u'], npdata['w']], axis=0)
            self.data = self.data.astype(np.float32)
            self.data = self.data.transpose(0, 1, 3, 2)  # [c, t, z, x]
        # size_t = self.data.shape[1] #192
        size_t = 192
        size_z = self.data.shape[2]
        size_x = self.data.shape[3]

        self.root = Node((0,0,0), (size_t,size_z,size_x), (size_t, size_z, size_x)) # initialize root node

        # self.depth = math.ceil(np.log2(max(size_t / self.crop_size[0], size_z / self.crop_size[1], size_x / self.crop_size[2]))) + self.n_min_vol_per_crop - 1
        self.depth = int(np.log2(max(size_t / self.crop_size[0], size_z / self.crop_size[1], size_x / self.crop_size[2]))+0.5) + self.n_min_vol_per_crop
        # self.depth = 6
        # build octree from the root
        if self.depth>0:
            self.root.is_leaf = False
            self.depth_now = self.depth_now + 1
            self.subdivide_node(self.root)
            self.depth_now = self.depth_now - 1
        else:
            print("Do not need subdivition. Octree has one node, i.e. the root.")
            self.leaves.append(self.root)

        time_end = time.time()
        print('time cost:', time_end - time_start, 's')

        # save octree to file
        print('save octree and leaf nodes...')
        f = open(os.path.join(self.data_dir, self.data_filename+'.octree_loss.pkl'), 'wb')
        f.write(pickle.dumps(self.root)) # save octree
        f = open(os.path.join(self.data_dir, self.data_filename+'.leaves_loss.pkl'), 'wb')
        f.write(pickle.dumps(self.leaves)) # save leaf 
        
    def subdivide_node(self, node):
        """
        Subdivide a node based on the volumetric data.
        Args:
            node: node to subdivide
        """

        zoom = [(node.size[0] / self.crop_size[0], 0), (node.size[1] / self.crop_size[1], 1), (node.size[2] / self.crop_size[2], 2)]
        zoom.sort(key=lambda zoom_item: zoom_item[0], reverse=True)

        # call different subdivied methods based on the zoom (i.e. the ration of node size to crop size in each dimmension)

        if zoom[0][0] >= zoom[2][0]*2 and zoom[1][0] >= zoom[2][0]*2:
            self.insert_node_4(node, zoom[0][1], zoom[1][1])
        elif zoom[0][0] >= zoom[2][0]*2:
            self.insert_node_2(node, zoom[0][1])
        elif zoom[2][0] >= 1:
            self.insert_node_8(node)
        else:
            node.is_leaf = True
            self.leaves.append(node)
            print('there is no need to subdivide this node.')

    def insert_node_2(self, node, index):
        """
        Subdivie a node with two nodes.
        Args:
            node: node to subdivide
            index: the index of the axis that needs to subdivide
        """

        # divide point on the index axis
        divide_point = node.vertex1[index] + node.size[index]//2 

        # calculate vertex1 and vertex2 of the new node on each branch
        for branch in range(2):
            if branch == 0:
                new_node_vertex1 = node.vertex1
                tmp = [node.vertex2[0], node.vertex2[1], node.vertex2[2]]
                tmp[index] = divide_point
                new_node_vertex2 = (tmp[0], tmp[1], tmp[2])
            elif branch == 1:
                tmp = [node.vertex1[0], node.vertex1[1], node.vertex1[2]]
                tmp[index] = divide_point
                new_node_vertex1 = (tmp[0], tmp[1], tmp[2])
                new_node_vertex2 = node.vertex2

            self.create_node(node, new_node_vertex1, new_node_vertex2)

    def insert_node_4(self, node, index1, index2):
        """
        Subdivie a node with four nodes.
        Args:
            node: node to subdivide
            index1: the first index of the axis that needs to subdivide
            index2: the second index of the axis that needs to subdivide
        """

        # divide point on the index axis
        divide_point1 = node.vertex1[index1] + node.size[index1]//2
        divide_point2 = node.vertex1[index2] + node.size[index2]//2

        for branch in range(4):
            if branch == 0:
                new_node_vertex1 = node.vertex1
                tmp = [node.vertex2[0], node.vertex2[1], node.vertex2[2]]
                tmp[index1] = divide_point1
                tmp[index2] = divide_point2
                new_node_vertex2 = (tmp[0], tmp[1], tmp[2])
            elif branch == 1:
                tmp = [node.vertex1[0], node.vertex1[1], node.vertex1[2]]
                tmp[index2] = divide_point2
                new_node_vertex1 = (tmp[0], tmp[1], tmp[2])
                tmp = [node.vertex2[0], node.vertex2[1], node.vertex2[2]]
                tmp[index1] = divide_point1
                new_node_vertex2 = (tmp[0], tmp[1], tmp[2])
            elif branch == 2:
                tmp = [node.vertex1[0], node.vertex1[1], node.vertex1[2]]
                tmp[index1] = divide_point1
                new_node_vertex1 = (tmp[0], tmp[1], tmp[2])
                tmp = [node.vertex2[0], node.vertex2[1], node.vertex2[2]]
                tmp[index2] = divide_point2
                new_node_vertex2 = (tmp[0], tmp[1], tmp[2])
            elif branch == 3:
                tmp = [node.vertex1[0], node.vertex1[1], node.vertex1[2]]
                tmp[index1] = divide_point1
                tmp[index2] = divide_point2
                new_node_vertex1 = (tmp[0], tmp[1], tmp[2])          
                new_node_vertex2 = node.vertex2
            # print(branch, new_node_vertex1, new_node_vertex2)

            self.create_node(node, new_node_vertex1, new_node_vertex2)

    def insert_node_8(self, node):
        """
        Subdivie a node with eight nodes.
        Args:
            node: node to subdivide
        """

        # divide point on all three axises
        divide_point = (node.vertex1[0] + node.size[0]//2, node.vertex1[1] + node.size[1]//2, node.vertex1[2] + node.size[2]//2)
        
        for branch in range(8):
            if branch == 0:
                # left down back
                new_node_vertex1 = node.vertex1
                new_node_vertex2 = divide_point
            elif branch == 1:
                # left down forwards
                new_node_vertex1 = (node.vertex1[0], node.vertex1[1], divide_point[2])
                new_node_vertex2 = (divide_point[0], divide_point[1], node.vertex2[2])
            elif branch == 2:
                # left up back
                new_node_vertex1 = (node.vertex1[0], divide_point[1], node.vertex1[2])
                new_node_vertex2 = (divide_point[0], node.vertex2[1], divide_point[2])

            elif branch == 3:
                # left up forward
                new_node_vertex1 = (node.vertex1[0], divide_point[1], divide_point[2])
                new_node_vertex2 = (divide_point[0], node.vertex2[1], node.vertex2[2])

            elif branch == 4:
                # right down back
                new_node_vertex1 = (divide_point[0], node.vertex1[1], node.vertex1[2])
                new_node_vertex2 = (node.vertex2[0], divide_point[1], divide_point[2])

            elif branch == 5:
                # right down forwards
                new_node_vertex1 = (divide_point[0], node.vertex1[1], divide_point[2])
                new_node_vertex2 = (node.vertex2[0], divide_point[1], node.vertex2[2])

            elif branch == 6:
                # right up back
                new_node_vertex1 = (divide_point[0], divide_point[1], node.vertex1[2])
                new_node_vertex2 = (node.vertex2[0], node.vertex2[1], divide_point[2])

            elif branch == 7:
                # right up forward
                new_node_vertex1 = divide_point
                new_node_vertex2 = node.vertex2

            self.create_node(node, new_node_vertex1, new_node_vertex2)

    def create_node(self, node, new_node_vertex1, new_node_vertex2):
        """
        Create a node using vertex1 and vertex2.
        Args:
            node: node to create
            new_node_vertex1: the left down back vertex
            new_node_vertex2: the right up forward vertex
        """

        # new_data = self.data[:,
        #                 new_node_vertex1[0] : new_node_vertex2[0],
        #                 new_node_vertex1[1] : new_node_vertex2[1],
        #                 new_node_vertex1[2] : new_node_vertex2[2]]  # [c, t, z, x]
        new_loss = self.loss[:,
                             int(new_node_vertex1[0]/4) : int(new_node_vertex2[0]/4),
                             int(new_node_vertex1[1]/8) : int(new_node_vertex2[1]/8),
                             int(new_node_vertex1[2]/8) : int(new_node_vertex2[2]/8)]

        new_loss_std = torch.mean(new_loss, axis=(1, 2, 3))
        # new_loss_std = torch.mean(new_loss)

        new_node = Node(new_node_vertex1, new_node_vertex2, (new_node_vertex2[0] - new_node_vertex1[0], new_node_vertex2[1] - new_node_vertex1[1], new_node_vertex2[2] - new_node_vertex1[2]))

        # arbitrary std is still big and depth_now does not reach the predefined depth, go further
        if self.depth_now < self.depth and (new_loss_std[0] > self.loss_std[0] or new_loss_std[1] > self.loss_std[1] or new_loss_std[2] > self.loss_std[2] or new_loss_std[3] > self.loss_std[3]):
        # if self.depth_now < self.depth and new_loss_std > self.loss_std:
            # print("depth now is: ", self.depth_now)
            # print("std of the new node: ", new_std)
            new_node.is_leaf = False
            node.branches.append(new_node)
            self.depth_now = self.depth_now + 1
            self.subdivide_node(new_node)
            self.depth_now = self.depth_now - 1 # backtracking
        else:
            # print("depth now is: ", self.depth_now)
            # print("std of the new node: ", new_std)
            self.leaves.append(new_node)
            node.branches.append(new_node)

    def __getitem__(self, key):
        """Get divided volumes of a data crop defined by the key.
        Args:
            key: tuple, (t_start, z_strat, x_start, t_len, z_len, x_len) of the cropped data sub-block.
        """

        # vertexes of the cropped data
        vertex1 = (key[0], key[1], key[2]) # vertex 1
        vertex2 = (key[0] + key[3], key[1] + key[4], key[2] + key[5]) # vertex 2

        rets = [] # the final list of all the valid nodes and parts of nodes, formatted as ret[[t_start, z_strat, x_start, t_len, z_len, x_len, area_factor], ...]
        octree_traverse_queue = SimpleQueue() # traverse for all octree to find valid nodes

        octree_traverse_queue.put(self.root)
        while not octree_traverse_queue.empty():
            current_node = octree_traverse_queue.get()
            x_intersect =                 (vertex1[0] <= current_node.vertex1[0] and vertex2[0] > current_node.vertex1[0]) or (vertex1[0] > current_node.vertex1[0] and vertex1[0] < current_node.vertex2[0])
            y_intersect = x_intersect and ((vertex1[1] <= current_node.vertex1[1] and vertex2[1] > current_node.vertex1[1]) or (vertex1[1] > current_node.vertex1[1] and vertex1[1] < current_node.vertex2[1]))
            z_intersect = y_intersect and ((vertex1[2] <= current_node.vertex1[2] and vertex2[2] > current_node.vertex1[2]) or (vertex1[2] > current_node.vertex1[2] and vertex1[2] < current_node.vertex2[2]))
            if z_intersect:
                if current_node.is_leaf:
                    rx1 = max(vertex1[0], current_node.vertex1[0])
                    ry1 = max(vertex1[1], current_node.vertex1[1])
                    rz1 = max(vertex1[2], current_node.vertex1[2])
                    rx2 = min(vertex2[0], current_node.vertex2[0])
                    ry2 = min(vertex2[1], current_node.vertex2[1])
                    rz2 = min(vertex2[2], current_node.vertex2[2])
                    rsizex = rx2 - rx1
                    rsizey = ry2 - ry1
                    rsizez = rz2 - rz1
                    ret = (rx1, ry1, rz1, rsizex, rsizey, rsizez, rsizex / current_node.size[0] * rsizey / current_node.size[1] * rsizez / current_node.size[2])

                    rets.append(ret)
                else:
                    for i in current_node.branches: octree_traverse_queue.put(i)
        
        return rets

class Node:

    def __init__(self, vertex1, vertex2, size):
        """
        Args:
            vertex1: tuple, (left_down_back_t, left_down_back_z, left_down_back_x) of the node.
            vertex2: tuple, (right_up_forward_t, right_up_forward_z, right_up_forward_x) of the node.
            size: tuple, (size_t, size_z, size_x) of the node.
        """

        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.size = size

        # all nodes are leaf nodes at first
        self.is_leaf = True
        
        # all nodes have empty branches at first
        self.branches = []

if __name__ == '__main__':

    # volumetric data and crop size
    data_size = (32, 16, 8) # (t, z, x)
    crop_size = (16, 16, 2) # (t, z, x)
    n_min_vol_per_crop = 1
    num_samp_pts_per_block = 8/n_min_vol_per_crop**3

    print('creat random data...')
    time_start = time.time()
    data = np.random.randn(4,data_size[0],data_size[1],data_size[2])
    # data = np.ones((4,data_size[0],data_size[1],data_size[2]))
    time_end = time.time()
    print('time cost:', time_end - time_start, 's')

    print('compute mean and std...')
    time_start = time.time()
    mean = np.mean(data, axis=(1, 2, 3))
    std = np.std(data, axis=(1, 2, 3))
    print('mean: ', mean)
    print('std: ', std)
    time_end = time.time()
    print('time cost:', time_end - time_start, 's') 

    my_octree = Octree_loss("./data", "test.npz", data, mean, std, crop_size, n_min_vol_per_crop)

    print('the number of leaves is: ', len(my_octree.leaves))
    print('the depth is: ', my_octree.depth)
    # for leaf in my_octree.leaves:
    #     print('--------------')
    #     print('size', leaf.size)
    #     print('vertex1', leaf.vertex1)
    #     print('vertex2', leaf.vertex2)
    #     print('--------------')

    # # load octree
    # my_octree = Octree("./data", "test.npz") 
    # print(my_octree.root.position)
    # print(my_octree.root.size)
    # print(len(my_octree.leaves))
    # print(my_octree.leaves[0].position)
    # print(my_octree.leaves[0].size)

    # give a query key, get a block list

    print('creat index of cropped data...')
    time_start = time.time()
    start_index = (np.random.randint(0, data_size[0]-crop_size[0]+1, 1)[0], np.random.randint(0, data_size[1]-crop_size[1]+1, 1)[0], np.random.randint(0, data_size[2]-crop_size[2]+1, 1)[0])
    query_key = (start_index[0], start_index[1], start_index[2], crop_size[0], crop_size[1], crop_size[2])
    # query_key = (0, 0, 0, 16, 16, 2)
    print(query_key)
    div_blocks_list = my_octree[query_key]
    time_end = time.time()
    print('time cost:', time_end - time_start, 's')

    print('the number of mini blocks is: ', len(div_blocks_list))
    sum_volume = 0
    # print('the list of divided blocks is:')
    for block in div_blocks_list:      
        # print(block)
        sum_volume += block[3]*block[4]*block[5]
    volume = crop_size[0]*crop_size[1]*crop_size[2]
    print('the volume sum of divided blocks is: ', sum_volume)
    print('the volume of this data crop is: ', volume)

    point_coord = []
    num_samp_points = 0
    for block in div_blocks_list: 
        print("block: ", block)
        samp_pts = np.random.rand(int(block[6]*num_samp_pts_per_block+0.5), 3) * ([block[3]-1, block[4]-1, block[5]-1]) + [block[0], block[1], block[2]]
        point_coord.extend(samp_pts)
        num_samp_points = num_samp_points + int(block[6]*num_samp_pts_per_block+0.5)
        print("number of sample points in this block: ", int(block[6]*num_samp_pts_per_block+0.5))
        # print("sample points: ", samp_pts)
        
    print("number of sample points in this data crop is: ", num_samp_points)
    # print(point_coord)
        