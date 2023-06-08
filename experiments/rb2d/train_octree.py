"""Training script for RB2 experiment.
"""
import argparse
import json
import os
from glob import glob
import numpy as np
from collections import defaultdict

np.set_printoptions(precision=4)

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

# import our modules
import sys

sys.path.append("../../src")
import train_utils as utils
from unet3d import UNet3d
from implicit_net import ImNet
from local_implicit_grid import query_local_implicit_grid
from nonlinearities import NONLINEARITIES
import dataloader_spacetime_octree as loader
from physics import get_rb2_pde_layer
import time
from octree_loss import Octree_loss
from scipy.stats import linregress
# pylint: disable=no-member

# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh = 1000000
prandtl = 1
gamma = 0.0125
use_continuity = True
# log_dir_name = "./log/treshold/"

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.cuda.set_device(0)


total_num_sample_points = 0


def loss_functional(loss_type):
    """Get loss function given function type names."""
    if loss_type == 'l1':
        return F.l1_loss
    if loss_type == 'l2':
        return F.mse_loss
    # else (loss_type == 'huber')
    return F.smooth_l1_loss

#vector padding
def padding(point,max_size):
    # point = np.pad(point, ((0,max_size-len(point)),(0,0)), 'constant')
    point = torch.tensor(point)
    if len(point) < max_size:
        point = F.pad(point, (0,0,0,max_size-len(point)), 'constant')
    point = point.unsqueeze(0)
    return point

def train(args, unet, imnet, train_loader, epoch, global_step, device,
          logger, writer, optimizer, pde_layer):
    """Training function."""
    unet.train()
    imnet.train()
    tot_loss = 0
    count = 0
    global total_num_sample_points
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)
    loss_func = loss_functional(args.reg_loss_type)
    for batch_idx, data_tensors in enumerate(train_loader):
        # send tensors to device

        # data_tensors = [t.to(device) for t in data_tensors]
        input_grid, point_coord, point_value = data_tensors
        input_grid = torch.from_numpy(input_grid).to(device)

        # Expanding with zero padding in numpy and combining into a tensor
        max_size = max(len(i) for i in point_coord)
        per_len = [len(i) for i in point_coord]

        point_coord_new_list = [padding(point,max_size) for point in point_coord]
        point_value_new_list = [padding(value,max_size) for value in point_value]

        #tensor concat
        point_coord_new = torch.cat(point_coord_new_list,0).to(device)
        point_value_new = torch.cat(point_value_new_list,0).to(device)

        optimizer.zero_grad()
        latent_grid = unet(input_grid)  # [batch, N, C, T, X, Y]
        # permute such that C is the last channel for local implicit grid query
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, N, T, X, Y, C]

        # define lambda function for pde_layer
        fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

        # update pde layer and compute predicted values + pde residues
        pde_layer.update_forward_method(fwd_fn)
        pred_value, residue_dict = pde_layer(point_coord_new, return_residue=True)

        #Restore pred_value and reset all dimensions that exceed it to 0.
        for i in range(point_coord_new.shape[0]):
            if per_len[i] < max_size:
                pred_value[i,per_len[i]:max_size,:] = 0

        # function value regression loss
        reg_loss = loss_func(pred_value, point_value_new)

        # pde residue loss
        pde_tensors = torch.stack([d for d in residue_dict.values()], dim=0)
        zero_pde_tensors = torch.zeros_like(pde_tensors)
        pde_loss = loss_func(pde_tensors, zero_pde_tensors)
        loss = args.alpha_reg * reg_loss + args.alpha_pde * pde_loss

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_value_(unet.module.parameters(), args.clip_grad)
        torch.nn.utils.clip_grad_value_(imnet.module.parameters(), args.clip_grad)

        optimizer.step()
        tot_loss += loss.item()
        # count += input_grid.size()[0]
        count += 1
        for i in per_len:
            if i > 2:
                total_num_sample_points += i
        if (batch_idx + 1) % args.log_interval == 0:
            # print('shape of pred_value per batch (b, n, c): ', list(pred_value.shape))
            # print('shape of pde_tensors per batch (b, n, c): ', list(pde_tensors.shape))
            # print("so far, the total number of sample points in this training is: ", total_num_sample_points)
            logger.info("so far, the total number of sample points in this training is: %d", total_num_sample_points)
            # logger log
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss Sum: {:.6f}\t"
                "Loss Reg: {:.6f}\tLoss Pde: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(input_grid), len(train_loader) * len(input_grid),
                           100. * (batch_idx + 1) / len(train_loader), loss.item(),
                           args.alpha_reg * reg_loss, args.alpha_pde * pde_loss))
            # tensorboard log
            writer.add_scalar('train/reg_loss_unweighted', reg_loss, global_step=int(epoch))
            writer.add_scalar('train/pde_loss_unweighted', pde_loss, global_step=int(epoch))
            writer.add_scalar('train/sum_loss', loss, global_step=int(epoch))
            writer.add_scalars('train/losses_weighted',
                               {"reg_loss": args.alpha_reg * reg_loss,
                                "pde_loss": args.alpha_pde * pde_loss,
                                "sum_loss": loss}, global_step=int(epoch))

        global_step += 1
    tot_loss /= count
    return tot_loss


def eval(args, unet, imnet, eval_loader, epoch, global_step, device,
         logger, writer, optimizer, pde_layer):
    """Eval function. Used for evaluating entire slices and comparing to GT."""
    unet.eval()
    imnet.eval()
    phys_channels = ["p", "b", "u", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)
    for data_tensors in eval_loader:
        # only need the first batch
        break
    # send tensors to device
    data_tensors = [t.to(device) for t in data_tensors]
    hres_grid, lres_grid = data_tensors
    latent_grid = unet(lres_grid)  # [batch, C, T, Z, X]
    # nb, nc, nt, nz, nx = lres_grid.shape
    nb, nc, nt, nz, nx = hres_grid.shape
    #scale index
    index = args.scale
    nt, nz, nx = int(nt/index), int(nz/index), int(nx/index)

    # permute such that C is the last channel for local implicit grid query
    latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

    # define lambda function for pde_layer
    fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

    # update pde layer and compute predicted values + pde residues
    pde_layer.update_forward_method(fwd_fn)

    # layout query points for the desired slices
    eps = 1e-6
    t_seq = torch.linspace(eps, 1 - eps, nt)  # temporal sequences
    z_seq = torch.linspace(eps, 1 - eps, nz)  # z sequences
    x_seq = torch.linspace(eps, 1 - eps, nx)  # x sequences
    query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq), axis=-1)  # [nt, nz, nx, 3]
    query_coord = query_coord.reshape([-1, 3]).to(device)  # [nt*nz*nx, 3]
    n_query = query_coord.shape[0]

    res_dict = defaultdict(list)
    n_iters = int(np.ceil(n_query / args.pseudo_batch_size))        #48

    for idx in range(n_iters):
        sid = idx * args.pseudo_batch_size
        eid = min(sid + args.pseudo_batch_size, n_query)
        query_coord_batch = query_coord[sid:eid]
        query_coord_batch = query_coord_batch[None].expand(*(nb, eid - sid, 3))  # [nb, eid-sid, 3]
        pred_value, residue_dict = pde_layer(query_coord_batch, return_residue=True)
        pred_value = pred_value.detach()

        for key in residue_dict.keys():
            residue_dict[key] = residue_dict[key].detach()
        for name, chan_id in zip(phys_channels, range(4)):
            res_dict[name].append(pred_value[..., chan_id])  # [b, pb]
        for name, val in residue_dict.items():
            res_dict[name].append(val[..., 0])  # [b, pb]

    for key in res_dict.keys():
        res_dict[key] = (torch.cat(res_dict[key], axis=1)
                         .reshape([nb, len(t_seq), len(z_seq), len(x_seq)]))

    # constructing loss distribution
    gt_value = hres_grid[0, :, ::index, ::index, ::index]
    res_value = torch.cat([res_dict['p'],res_dict['b'],res_dict['u'],res_dict['w']],axis =0)
    loss_abs = torch.abs(res_value - gt_value)
    loss_std = torch.std(loss_abs, axis=(1, 2, 3))
    if args.div_method == "mean":
        loss_std = torch.mean(loss_abs, axis=(1, 2, 3))

    #create new tree
    Octree_loss('./data', 'rb2d_ra1e6_s102.npz',loss=loss_abs, loss_std=loss_std)

def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--slope_num", type=int,default=10, help="compute loss numbers(default: 10).")
    parser.add_argument("--recon_min", type=int, default=10,help="minimum reconstruction interval(default: 10)")
    parser.add_argument("--slope_treshold", type=int, default=0,help="the slope treshold of the loss.(default: 0)")
    parser.add_argument("--scale", type=int, default=2,help="scale_index")
    parser.add_argument("--div_method", type=str, default="std", choices=["mean", "std"],help="divide tree by mean(std).(default:std)")
    parser.add_argument("--batch_size_per_gpu", type=int, default=10, metavar="N",
                        help="input batch size for training (default: 10)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--pseudo_epoch_size", type=int, default=2000, metavar="N",
                        help="number of samples in an pseudo-epoch. (default: 3000)")
    parser.add_argument("--lr", type=float, default=1e-2, metavar="R",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--data_folder", type=str, default="./data",
                        help="path to data folder (default: ./data)")
    parser.add_argument("--train_data", type=str, default="rb2d_ra1e6_s102.npz",
                        help="name of training data (default: rb2d_ra1e6_s102.npz)")
    parser.add_argument("--eval_data", type=str, default="rb2d_ra1e6_s42.npz",
                        help="name of training data (default: rb2d_ra1e6_s42.npz)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log_dir", type=str,default="" ,help="log directory for run")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint if resume is needed")
    parser.add_argument("--nt", default=16, type=int, help="resolution of high res crop in t.")
    parser.add_argument("--nx", default=128, type=int, help="resolution of high res crop in x.")
    parser.add_argument("--nz", default=128, type=int, help="resolution of high res crop in z.")
    parser.add_argument("--downsamp_t", default=4, type=int,
                        help="down sampling factor in t for low resolution crop.")
    parser.add_argument("--downsamp_xz", default=8, type=int,
                        help="down sampling factor in x and z for low resolution crop.")
    parser.add_argument("--n_samp_pts_per_crop", default=512, type=int,
                        help="number of sample points to draw per crop.")
    parser.add_argument("--lat_dims", default=32, type=int, help="number of latent dimensions.")
    parser.add_argument("--unet_nf", default=16, type=int,
                        help="number of base number of feature layers in unet.")
    parser.add_argument("--unet_mf", default=256, type=int,
                        help="a cap for max number of feature layers throughout the unet.")
    parser.add_argument("--imnet_nf", default=32, type=int,
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--reg_loss_type", default="l1", type=str,
                        choices=["l1", "l2", "huber"],
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--alpha_reg", default=1., type=float, help="weight of regression loss.")
    parser.add_argument("--alpha_pde", default=gamma, type=float, help="weight of pde residue loss.")
    parser.add_argument("--num_log_images", default=2, type=int, help="number of images to log.")
    parser.add_argument("--pseudo_batch_size", default=1024, type=int,
                        help="size of pseudo batch during eval.")
    parser.add_argument("--normalize_channels", dest='normalize_channels', action='store_true')
    parser.add_argument("--no_normalize_channels", dest='normalize_channels', action='store_false')
    parser.set_defaults(normalize_channels=True)
    parser.add_argument("--lr_scheduler", dest='lr_scheduler', action='store_true')
    parser.add_argument("--no_lr_scheduler", dest='lr_scheduler', action='store_false')
    parser.set_defaults(lr_scheduler=True)
    parser.add_argument("--clip_grad", default=1., type=float,
                        help="clip gradient to this value. large value basically deactivates it.")
    parser.add_argument("--lres_filter", default='none', type=str,
                        help=("type of filter for generating low res input data. "
                              "choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'."))
    parser.add_argument("--lres_interp", default='linear', type=str,
                        help=("type of interpolation scheme for generating low res input data."
                              "choice of 'linear', 'nearest'"))
    parser.add_argument('--rayleigh', type=float, default=rayleigh,
                        help='Simulation Rayleigh number.')
    parser.add_argument('--prandtl', type=float, default=prandtl,
                        help='Simulation Prandtl number.')
    parser.add_argument('--nonlin', type=str, default='softplus', choices=list(NONLINEARITIES.keys()),
                        help='Nonlinear activations for continuous decoder.')
    parser.add_argument('--use_continuity', type=str2bool, nargs='?', default=use_continuity, const=True,
                        help='Whether to enforce continuity equation (mass conservation) or not')

    args = parser.parse_args()
    return args


def main():
    start = time.perf_counter()  # count time

    args = get_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    # adjust batch size based on the number of gpus available
    args.batch_size = int(torch.cuda.device_count()) * args.batch_size_per_gpu
    # log_dir = "./log/aim{}/min_epoch{}".format(args.slope_num,args.aim_epoch)

    os.makedirs(args.log_dir, exist_ok=True)
    filenames_to_snapshot = glob("*.py") + glob("*.sh")
    utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), 'w') as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))

    # random seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloaders
    trainset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename=args.train_data,
        nx=args.nx, nz=args.nz, nt=args.nt, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, return_hres=False,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp,div_method = args.div_method
    )
    evalset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename=args.train_data,
        nx=512, nz=128, nt=192, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, return_hres=True,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp
    )

    train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.pseudo_epoch_size)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              sampler=train_sampler, collate_fn=loader.pad_num_pts, **kwargs)
    eval_loader = DataLoader(evalset, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    # setup model
    unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=trainset.scale_lres,
                  nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf,
                  activation=NONLINEARITIES[args.nonlin])
    all_model_params = list(unet.parameters()) + list(imnet.parameters())

    if args.optim == "sgd":
        optimizer = optim.SGD(all_model_params, lr=args.lr)
    else:
        optimizer = optim.Adam(all_model_params, lr=args.lr)

    start_ep = 0
    global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf

    if args.resume:
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        unet.load_state_dict(resume_dict["unet_state_dict"])
        imnet.load_state_dict(resume_dict["imnet_state_dict"])
        optimizer.load_state_dict(resume_dict["optim_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    unet = nn.DataParallel(unet)
    unet.to(device)
    imnet = nn.DataParallel(imnet)
    imnet.to(device)

    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    logger.info("{}(unet) + {}(imnet) paramerters in total".format(model_param_count(unet),
                                                                   model_param_count(imnet)))

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest.pth.tar")

    # get pdelayer for the RB2 equations
    if args.normalize_channels:
        mean = trainset.channel_mean
        std = trainset.channel_std
    else:
        mean = std = None
    pde_layer = get_rb2_pde_layer(mean=mean, std=std,
                                  t_crop=args.nt * 0.125, z_crop=args.nz * (1. / 128), x_crop=args.nx * (1. / 128),
                                  prandtl=args.prandtl, rayleigh=args.rayleigh,
                                  use_continuity=args.use_continuity)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    #saving loss list
    loss_info = []
    tmp_epoch = -args.recon_min
    x = [i for i in range(args.slope_num)]

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        loss = train(args, unet, imnet, train_loader, epoch, global_step, device, logger, writer,
                     optimizer, pde_layer)

        #rebuild loss tree
        loss_info.append(loss)
        slope = -10
        if len(loss_info) >=args.slope_num:
            y = loss_info[-args.slope_num:]
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
        if slope >= args.slope_treshold and epoch-tmp_epoch>=args.recon_min:
            tmp_epoch = epoch
            octree_name = os.path.join(args.data_folder, args.train_data + '.octree_loss.pkl')
            leaves_name = os.path.join(args.data_folder, args.train_data + '.leaves_loss.pkl')
            if os.path.exists(octree_name):
                os.remove(octree_name)
                os.remove(leaves_name)
            logger.info("create new octree_loss")
            eval(args, unet, imnet, eval_loader, epoch, global_step, device, logger, writer, optimizer,
            pde_layer)

            #create new dataloader by loss tree
            trainset = loader.RB2DataLoader(
                data_dir=args.data_folder, data_filename=args.train_data,
                nx=args.nx, nz=args.nz, nt=args.nt, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
                downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
                normalize_output=args.normalize_channels, return_hres=False,
                lres_filter=args.lres_filter, lres_interp=args.lres_interp,
                loss_tree=True
            )
            train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.pseudo_epoch_size)
            train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                      sampler=train_sampler, collate_fn=loader.pad_num_pts, **kwargs)

        if args.lr_scheduler:
            scheduler.step(loss)
        if loss < tracked_stats:
            tracked_stats = loss
            is_best = True
        else:
            is_best = False

        utils.save_checkpoint({
            "epoch": epoch,
            "unet_state_dict": unet.module.state_dict(),
            "imnet_state_dict": imnet.module.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "tracked_stats": tracked_stats,
            "global_step": global_step,
        }, is_best, epoch, checkpoint_path, "_pdenet", logger)

    end = time.perf_counter()
    logger.info("The best loss in this training is: %d", tracked_stats)
    logger.info("The total number of sample points in this training is: %d", total_num_sample_points)
    logger.info("Total time = %d second", round(end - start))
    with open(args.log_dir+"/loss.txt", "w") as f:
        for los in loss_info:
            f.write("{}\n".format(los))

if __name__ == "__main__":
    main()
