"""RB2 Experiment Dataloader"""
import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import warnings
# pylint: disable=too-manz-arguments, too-manz-instance-attributes, too-manz-locals

import sys
sys.path.append("../src")

from divid_octree import Octree
from divid_octree_loss import Octree_loss
from torch.utils.data import DataLoader, RandomSampler

class RB2DataLoader(Dataset):
    """Pytorch Dataset instance for loading Rayleigh Bernard 2D dataset.

    Loads a 2d space + time cubic cutout from the whole simulation.
    """
    def __init__(self, data_dir="./data", data_filename="rb2d_ra1e6_s102.npz",
                 nx=128, nz=128, nt=16, n_samp_pts_per_crop=1024,
                 downsamp_xz=4, downsamp_t=4, normalize_output=False, normalize_hres=False,
                 return_hres=False, lres_filter='none', lres_interp='linear',loss_tree=False,div_method="std"):
        """

        Initialize DataSet
        Args:
          data_dir: str, path to the dataset folder, default="./data"
          data_filename: str, name of the dataset file, default="rb2d_ra1e6_s42.npz"
          nx: int, number of 'pixels' in x dimension for high res dataset.
          nz: int, number of 'pixels' in z dimension for high res dataset.
          nt: int, number of timesteps in time for high res dataset.
          n_samp_pts_per_crop: int, number of sample points to return per crop.
          downsamp_xz: int, downsampling factor for the spatial dimensions.
          downsamp_t: int, downsampling factor for the temporal dimension.
          normalize_output: bool, whether to normalize the range of each channel to [0, 1].
          normalize_hres: bool, normalize high res grid.
          return_hres: bool, whether to return the high-resolution data.
          lres_filter: str, filter to apply on original high-res image before interpolation.
                       choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'.
          lres_interp: str, interpolation scheme for generating low res.
                       choice of 'linear', 'nearest'.
          loss_tree： bool, whether to return loss tree
          div_method: str, divide tree by std or mean, default="std"
        """
        self.data_dir = data_dir
        self.data_filename = data_filename
        self.nx_hres = nx
        self.nz_hres = nz
        self.nt_hres = nt
        self.nx_lres = int(nx/downsamp_xz)
        self.nz_lres = int(nz/downsamp_xz)
        self.nt_lres = int(nt/downsamp_t)
        self.n_samp_pts_per_crop = n_samp_pts_per_crop
        self.downsamp_xz = downsamp_xz
        self.downsamp_t = downsamp_t
        self.normalize_output = normalize_output
        self.normalize_hres = normalize_hres
        self.return_hres = return_hres
        self.lres_filter = lres_filter
        self.lres_interp = lres_interp

        self.num_samp_pts = n_samp_pts_per_crop

        self.num_samp_pts_per_block = 0
        self.n_min_vol_per_crop = 1
        self.tot_num_samp_pts = 0
        self.loss_tree = loss_tree
        self.div_method = div_method

        # warn about median filter
        if lres_filter == 'median':
            warnings.warn("the median filter is very slow...", RuntimeWarning)

        # concatenating pressure, temperature, x-velocity, and z-velocity as a 4 channel array: pbuw
        # shape: (4, 200, 512, 128)
        npdata = np.load(os.path.join(self.data_dir, self.data_filename))
        self.data = np.stack([npdata['p'], npdata['b'], npdata['u'], npdata['w']], axis=0)
        self.data = self.data.astype(np.float32)
        self.data = self.data.transpose(0, 1, 3, 2)  # [c, t, z, x]
        nc_data, nt_data, nz_data, nx_data = self.data.shape

        # assert nx, nz, nt are viable
        if (nx > nx_data) or (nz > nz_data) or (nt > nt_data):
            raise ValueError('Resolution in each spatial temporal dimension x ({}), z({}), t({})'
                             'must not exceed dataset limits x ({}) z ({}) t ({})'.format(
                                 nx, nz, nt, nx_data, nz_data, nt_data))
        if (nt % downsamp_t != 0) or (nx % downsamp_xz != 0) or (nz % downsamp_xz != 0):
            print('nt:%d, downsamp_t:%d, nx:%d, nz:%d, downsamp_xz:%d\n' % (nt,downsamp_t,nx, nz, downsamp_xz))
            raise ValueError('nx, nz and nt must be divisible by downsamp factor.')

        self.nx_start_range = np.arange(0, nx_data-nx+1)
        self.nz_start_range = np.arange(0, nz_data-nz+1)
        self.nt_start_range = np.arange(0, nt_data-nt+1)
        self.rand_grid = np.stack(np.meshgrid(self.nt_start_range,
                                              self.nz_start_range,
                                              self.nx_start_range, indexing='ij'), axis=-1)
        # (xaug, zaug, taug, 3)
        self.rand_start_id = self.rand_grid.reshape([-1, 3])
        self.scale_hres = np.array([self.nt_hres, self.nz_hres, self.nx_hres], dtype=np.int32)
        self.scale_lres = np.array([self.nt_lres, self.nz_lres, self.nx_lres], dtype=np.int32)

        # compute channel-wise mean and std
        self._mean = np.mean(self.data, axis=(1, 2, 3))
        self._std = np.std(self.data, axis=(1, 2, 3))

        self.num_samp_pts_per_block = self.n_samp_pts_per_crop/(self.n_min_vol_per_crop+1)**3
        # self.num_samp_pts_per_block = 64
        # print("the number of sample points per block is: ", self.num_samp_pts_per_block)
        # build octree for sampling
        if self.loss_tree == False:
            self.octree = Octree(self.data_dir, self.data_filename, self.data, self._mean, self._std, (self.nt_hres, self.nz_hres, self.nx_hres), self.n_min_vol_per_crop,self.div_method)
        if self.loss_tree == True:
            self.octree = Octree_loss(self.data_dir, self.data_filename)
        # print('the number of leaves is: ', len(self.octree.leaves))
        # print('the depth is: ', self.octree.depth)

    def __len__(self):
        return self.rand_start_id.shape[0]

    def filter(self, signal):
        """Filter a given signal with a choice of filter type (self.lres_filter).
        """
        signal = signal.copy()
        filter_size = [1, self.downsamp_t*2-1, self.downsamp_xz*2-1, self.downsamp_xz*2-1]

        if self.lres_filter == 'none' or (not self.lres_filter):
            output = signal
        elif self.lres_filter == 'gaussian':
            sigma = [0, int(self.downsamp_t/2), int(self.downsamp_xz/2), int(self.downsamp_xz/2)]
            output = ndimage.gaussian_filter(signal, sigma=sigma)
        elif self.lres_filter == 'uniform':
            output = ndimage.uniform_filter(signal, size=filter_size)
        elif self.lres_filter == 'median':
            output = ndimage.median_filter(signal, size=filter_size)
        elif self.lres_filter == 'maximum':
            output = ndimage.maximum_filter(signal, size=filter_size)
        else:
            raise NotImplementedError(
                "lres_filter must be one of none/gaussian/uniform/median/maximum")
        return output

    def __getitem__(self, idx):
        """Get the random cutout data cube corresponding to idx.

        Args:
          idx: int, index of the crop to return. must be smaller than len(self).

        Returns:
          space_time_crop_hres (*optional): array of shape [4, nt_hres, nz_hres, nx_hres],
          where 4 are the phys channels pbuw.
          space_time_crop_lres: array of shape [4, nt_lres, nz_lres, nx_lres], where 4 are the phys
          channels pbuw.
          point_coord: array of shape [n_samp_pts_per_crop, 3], where 3 are the t, x, z dims.
                       CAUTION - point_coord are normalized to (0, 1) for the relative window.
          point_value: array of shape [n_samp_pts_per_crop, 4], where 4 are the phys channels pbuw.
        """
        t_id, z_id, x_id = self.rand_start_id[idx]
        space_time_crop_hres = self.data[:,
                                         t_id:t_id+self.nt_hres,
                                         z_id:z_id+self.nz_hres,
                                         x_id:x_id+self.nx_hres]  # [c, t, z, x]

        # create low res grid from hi res space time crop
        # apply filter
        space_time_crop_hres_fil = self.filter(space_time_crop_hres)

        interp = RegularGridInterpolator(
            (np.arange(self.nt_hres), np.arange(self.nz_hres), np.arange(self.nx_hres)),
            values=space_time_crop_hres_fil.transpose(1, 2, 3, 0), method=self.lres_interp)

        lres_coord = np.stack(np.meshgrid(np.linspace(0, self.nt_hres-1, self.nt_lres),
                                          np.linspace(0, self.nz_hres-1, self.nz_lres),
                                          np.linspace(0, self.nx_hres-1, self.nx_lres),
                                          indexing='ij'), axis=-1)
        space_time_crop_lres = interp(lres_coord).transpose(3, 0, 1, 2)  # [c, t, z, x]

        # create random point samples within space time crop

        # octree-based sampling: the number of point samples = n_samp_pts_per_crop * sampling factor
        index = (t_id, z_id, x_id, self.nt_hres, self.nz_hres, self.nx_hres) # query key
        # print("query key is: ", index)

        # # get samp_factor from octree. samp_factor: float
        # samp_factor = self.octree[index] 
        # self.num_samp_pts = self.n_samp_pts_per_crop * samp_factor
        # point_coord = np.random.rand(self.num_samp_pts, 3) * (self.scale_hres - 1)
        # # octree-based sampling: the number of point samples = n_samp_pts_per_crop * sampling factor

        # get division result from octree and sample points from each data block. div_result: list.
        div_result = self.octree[index]
        # print('the number of mini blocks is: ', len(div_result))
        sum_volume = 0
        for block in div_result:      
            # print(block)
            sum_volume += block[3]*block[4]*block[5]
        volume = self.nt_hres*self.nz_hres*self.nx_hres
        # print('the volume sum of divided blocks is: ', sum_volume)
        # print('the volume of this data crop is: ', volume)
        point_coord = []
        num_samp_points = 0
        for block in div_result:
            # sam_pts = np.random.rand(int(block[6]*self.num_samp_pts_per_block+0.5), 3) * ([block[3]-1, block[4]-1, block[5]-1]) + [block[0], block[1], block[2]]
            samp_min = int(block[6] * self.num_samp_pts_per_block + 0.5)
            sam_pts = np.random.rand(samp_min, 3) * ([self.nt_hres - 1, self.nz_hres - 1, self.nx_hres - 1])
            point_coord.extend(sam_pts)
            # print("number of sample points in this data block is: ", int(block[6]*self.num_samp_pts_per_block+0.5))
            # print(block[6])
            num_samp_points +=int(block[6]*self.num_samp_pts_per_block+0.5)
        # print("number of sample points in this data crop is: ", num_samp_points)
        self.tot_num_samp_pts = num_samp_points
        # print("total number of sample points in this crop is: ", self.tot_num_samp_pts)
        # print(point_coord)
        
               
        point_value = interp(point_coord)
        point_coord = point_coord / (self.scale_hres - 1)

        if self.normalize_output:
            space_time_crop_lres = self.normalize_grid(space_time_crop_lres)
            point_value = self.normalize_points(point_value)
        if self.normalize_hres:
            space_time_crop_hres = self.normalize_grid(space_time_crop_hres)

        return_tensors = [space_time_crop_lres, point_coord, point_value]

        # cast everything to float32
        return_tensors = [t.astype(np.float32) for t in return_tensors]

        if self.return_hres:
            return_tensors = [space_time_crop_hres] + [space_time_crop_lres]
            return_tensors = [t.astype(np.float32) for t in return_tensors]
        return tuple(return_tensors)

    @property
    def channel_mean(self):
        """channel-wise mean of dataset."""
        return self._mean

    @property
    def channel_std(self):
        """channel-wise mean of dataset."""
        return self._std

    @staticmethod
    def _normalize_array(array, mean, std):
        """normalize array (np or torch)."""
        if isinstance(array, torch.Tensor):
            dev = array.device
            std = torch.tensor(std, device=dev)
            mean = torch.tensor(mean, device=dev)
        return (array - mean) / std

    @staticmethod
    def _denormalize_array(array, mean, std):
        """normalize array (np or torch)."""
        if isinstance(array, torch.Tensor):
            dev = array.device
            std = torch.tensor(std, device=dev)
            mean = torch.tensor(mean, device=dev)
        return array * std + mean

    def normalize_grid(self, grid):
        """Normalize grid.

        Args:
          grid: np array or torch tensor of shape [4, ...], 4 are the num. of phys channels.
        Returns:
          channel normalized grid of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(grid.shape)
        mean_bc = self.channel_mean[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        std_bc = self.channel_std[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        return self._normalize_array(grid, mean_bc, std_bc)


    def normalize_points(self, points):
        """Normalize points.

        Args:
          points: np array or torch tensor of shape [..., 4], 4 are the num. of phys channels.
        Returns:
          channel normalized points of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(points.shape)
        mean_bc = self.channel_mean[(None,)*(g_dim-1)]  # unsqueeze from the front
        std_bc = self.channel_std[(None,)*(g_dim-1)]  # unsqueeze from the front
        return self._normalize_array(points, mean_bc, std_bc)

    def denormalize_grid(self, grid):
        """Denormalize grid.

        Args:
          grid: np array or torch tensor of shape [4, ...], 4 are the num. of phys channels.
        Returns:
          channel denormalized grid of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(grid.shape)
        mean_bc = self.channel_mean[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        std_bc = self.channel_std[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        return self._denormalize_array(grid, mean_bc, std_bc)


    def denormalize_points(self, points):
        """Denormalize points.

        Args:
          points: np array or torch tensor of shape [..., 4], 4 are the num. of phys channels.
        Returns:
          channel denormalized points of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(points.shape)
        mean_bc = self.channel_mean[(None,)*(g_dim-1)]  # unsqueeze from the front
        std_bc = self.channel_std[(None,)*(g_dim-1)]  # unsqueeze from the front
        return self._denormalize_array(points, mean_bc, std_bc)

def pad_num_pts(batch_data):
    """Collate data within a batch.

    Args:
        batch_data: list, [[lowres_input_batch, point_coords, point_values], ...]
    Returns:
        list, [lowres_input_batch, point_coords, point_values]
    """

    batch_len = len(batch_data)

    lowres_input_batch = []
    point_coords = []
    point_values = []
    for i in range(batch_len):
        lowres_input_batch.append(batch_data[i][0])
        point_coords.append(batch_data[i][1])
        point_values.append(batch_data[i][2])

    lowres_input_batch = np.stack(lowres_input_batch, axis=0)
    ret = [lowres_input_batch, point_coords, point_values]

    return tuple(ret)

def test_pad_num_pts(batch_data):
    """Collate data within a batch.

    Args:
        batch_data: list, [[hires_input_batch, lowres_input_batch, point_coords, point_values], ...]
    Returns:
        list, [hires_input_batch, lowres_input_batch, point_coords, point_values]
    """

    batch_len = len(batch_data)
    print(type(batch_data))

    hires_input_batch = []
    lowres_input_batch = []
    point_coords = []
    point_values = []
    for i in range(batch_len):
        hires_input_batch.append(batch_data[i][0])
        lowres_input_batch.append(batch_data[i][1])
        point_coords.append(batch_data[i][2])
        point_values.append(batch_data[i][3])

    hires_input_batch = np.stack(hires_input_batch, axis=0)
    lowres_input_batch = np.stack(lowres_input_batch, axis=0)
    ret = [hires_input_batch, lowres_input_batch, point_coords, point_values]

    return tuple(ret)

if __name__ == '__main__':
    ### example for using the data loader
    dataset = RB2DataLoader(nx=512, nz=128, nt=200,n_samp_pts_per_crop=1000, downsamp_t=4, downsamp_xz=8, return_hres=True)
    print("length of dataset is: ", len(dataset))
    # ret = dataset[0]
    # print(ret[0].shape)
    # print(ret[1].shape)
    # print(ret[2].shape)
    # print(ret[3].shape)

    # lres_crop, point_coord, point_value = data_loader[61234]
    # import matplotlib.pyplot as plt
    # plt.scatter(point_coord[:, 1], point_coord[:, 2], c=point_value[:, 0])
    # plt.colorbar()
    # plt.show()
    # plt.imshow(lres_crop[0, :, :, 0].T, origin='lower'); plt.show()
    # plt.imshow(lres_crop[1, :, :, 0].T, origin='lower'); plt.show()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")

    batch_size = 2
    num_samples = 2
    total_num_sample_points = 0
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=num_samples)
    data_batches = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=0, collate_fn=test_pad_num_pts)

    for batch_idx, (hires_input_batch, lowres_input_batch, point_coords, point_values) in enumerate(data_batches):
        # print("Reading batch #{}:\t with lowres inputs of size {}, sample coord of size {}, sampe val of size {}"
        #       .format(batch_idx+1, list(lowres_input_batch.shape),  list(point_coords.shape), list(point_values.shape)))
        print("Reading batch #{}:\t with hires inputs of size {}, lowres inputs of size {}, sample coord of size {}, sample val of size {}"
              .format(batch_idx+1, list(hires_input_batch.shape), list(lowres_input_batch.shape),  len(point_coords), len(point_values)))

        print(type(hires_input_batch))
        hires_input_batch = torch.from_numpy(hires_input_batch).to(device)
        print(type(hires_input_batch))
        print(hires_input_batch.is_cuda)

        for i in range(len(point_coords)):
            total_num_sample_points += point_coords[i].shape[0]

            hires_input_batch_i = torch.unsqueeze(hires_input_batch[i], 0)
            print(hires_input_batch_i.shape)
            print(hires_input_batch_i.is_cuda)

            point_coord_i = np.expand_dims(point_coords[i], axis=0)
            point_coord_i = torch.from_numpy(point_coord_i).to(device)
            print(point_coord_i.is_cuda)
        
        # for i in range(len(point_coords)):
        #     print(np.expand_dims(hires_input_batch[i], axis=0).shape)
        #     print(np.expand_dims(point_coords[i], axis=0).shape)
        #     print(np.expand_dims(point_values[i], axis=0).shape)
        # if batch_idx > 5:
        #     break
    print("total number of sample points in this training is: ", total_num_sample_points)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax1 = fig.add_subplots(121)
    # ax2 = fig.add_subplots(122)
    # ax1.imshow(hires_input_batch[0, 0, 2])
    # ax2.imshow(lowres_input_batch[0, 0, 8])
    # plt.show()
