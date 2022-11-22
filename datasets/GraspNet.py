# -*- coding:utf-8 -*-#
# Common libs
import os
import time
import numpy as np
import pickle
import torch
import math
import random
import glob
import xml.dom.minidom


# OS functions
from os import listdir
from os.path import exists, join

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info

from datasets.common import grid_subsampling
from utils.config import bcolors

# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class GraspNetDataset(PointCloudDataset):

    def __init__(self, config, mode):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'GraspNet')

        ############
        # Parameters
        ############

        # Dataset folder
        if config.specify_dataset:
            config.use_seg_data = config.dataset_names[mode][1]
            config.object_label_dir = config.dataset_names[mode][2]
        if not os.path.exists(config.object_label_dir):
            print(config.object_label_dir)
            raise ('The specified folder(object_label_dir) does not exist')
        self.path = config.object_label_dir
        self.grasp_pkl_path = join(self.path, 'about_grasp')
        if not os.path.exists(self.grasp_pkl_path):
            os.makedirs(self.grasp_pkl_path)

        # Type of task conducted on this dataset
        self.dataset_task = 'regression'
        self.gripper_dim = 4+3+4
        self.world_dim = 4 + 3
        self.label_max = 198  # an issue rooted in history
        self.use_seg_data = config.use_seg_data

        # Update  data task in configuration
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.mode = mode

        object_names = self.get_object_information(config.object_label_dir)
        random.seed(config.seed)
        random.shuffle(object_names)
        self.object_names = object_names

        #############
        # Load models
        #############

        if 0 < self.config.first_subsampling_dl <= 0.001:
            raise ValueError('subsampling_parameter too low (should be over 1 mm')

        self.input_points, self.input_features, self.input_labels, self.num_samples, self.grasp_obj_name = self.load_subsampled_clouds(self.config.specify_dataset)

        if self.mode == 'train':
            self.epoch_n = config.epoch_steps * config.batch_num
        else:
            self.epoch_n = min(self.num_samples * self.label_max, config.validation_size * config.batch_num)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return self.num_samples * self.label_max

    def __getitem__(self, idx_list):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        ###################
        # Gather batch data
        ###################

        tp_list = []
        tf_list = []
        tl_list = []
        ti_list = []
        t_list = []
        R_list = []

        for p_ij in idx_list:

            p_i = p_ij // self.label_max
            p_j = p_ij % self.label_max

            # Get points and labels
            points = self.input_points[p_i].astype(np.float32)
            if self.use_seg_data:
                label = self.input_labels[p_i][0].reshape(1, -1)
                graspparts = self.input_features[p_i].astype(np.float32)[:, :16]
            else:
                raise ('please use_seg_data !!!')

            # Data augmentation
            points, label, tran, R = self.augmentation_grasp(points, label, is_norm=True, verbose=False, is_aug=self.config.is_aug) # incomplete

            # Stack batch
            tp_list += [points]
            tf_list += [graspparts]
            tl_list += [label]
            ti_list += [p_i]
            t_list += [tran]
            R_list += [R]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(tp_list, axis=0)
        labels = np.array(tl_list, dtype=np.float32)
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
        trans = np.array(t_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, stacked_points))
        elif self.config.in_features_dim == 20:
            stacked_grasppart = np.concatenate(tf_list, axis=0)
            stacked_features = np.hstack((stacked_features, stacked_points, stacked_grasppart))
        elif self.config.in_features_dim == 16:
            stacked_features = np.concatenate(tf_list, axis=0)  # # stacked_grasppart
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 , 16 and 20')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.graspnet_inputs(stacked_points,
                                          stacked_features,
                                          labels,
                                          stack_lengths)

        # Add scale and rotation for testing
        input_list += [trans, rots, model_inds]

        return input_list

    def get_object_information(self, object_dir):  # 遍历物体数据集文件夹，获取所有物体的名字
        object_names = []
        for name in sorted(os.listdir(object_dir)):
            if os.path.splitext(name)[1] == '.npy':
                object_names.append(name[:-24])
        return object_names

    def load_subsampled_clouds(self, specify_dataset):

        # Restart timer
        t0 = time.time()

        split = self.mode
        print('\nLoading {:s} points subsampled at {:.3f}'.format(split, self.config.first_subsampling_dl))
        if specify_dataset:
            filename_pkl = join(self.grasp_pkl_path, self.config.dataset_names[split][0])
            use_seg_data = self.config.dataset_names[split][1]
        else:
            raise ValueError('please use specify_dataset')
        print(self.mode, ' - filename is', filename_pkl)

        if exists(filename_pkl):
            with open(filename_pkl, 'rb') as file:
                input_points, input_features, input_labels, num_samples, grasp_obj_name = pickle.load(file)

        else:
            # Initialize containers
            input_points = []
            input_features = []
            input_labels = []
            grasp_obj_name = []

            # Advanced display
            progress_n = 100
            label_max = self.label_max
            fmt_str = '[{:<' + str(progress_n) + '}] {:5.4f}%'

            frame_id = 0
            for labeled_data in sorted(os.listdir(self.config.object_label_dir)):  # # 在标注的抓取分割数据集中遍历
                if labeled_data.endswith('.npy'):  # os.path.splitext(name)[1] == '.xml':
                    labeled_verts = np.load(join(self.config.object_label_dir, labeled_data))
                    verts = labeled_verts[:, :3].astype(np.float32) / 1000.0
                    grasp_label = labeled_verts[:, 3:].astype(np.float32)
                    if ((grasp_label!=1) & (grasp_label!=0)).any():
                        raise ValueError('grasp_label error in {}'.format(labeled_data))

                    # Subsample them
                    if self.config.first_subsampling_dl > 0:
                        points, features = grid_subsampling(verts,
                                                  features=grasp_label,  # # 分割特征也参与进来
                                                  sampleDl=self.config.first_subsampling_dl)
                        # features[(features != 1) & (features != 0)] = 1
                        # # 降采样去噪和恢复
                        features_mask0 = (features != 1) & (features != 0)
                        features_mask1 = features == np.max(features, axis=1)[:, None]
                        features[features_mask0 & features_mask1] = 1
                        features[features_mask0 & ~features_mask1] = 0

                        if ((features != 1) & (features != 0)).any():
                            raise ValueError('feature error in {}'.format(labeled_data))
                    else:
                        points = verts
                        features = grasp_label

                    frame_id += 1

                    # Add to list
                    input_points += [points]
                    input_features += [features]
                    name = labeled_data[:-24]
                    grasp_obj_name.append(name)
                    label_array = np.zeros((1, self.gripper_dim))
                    input_labels += [label_array]

            print('', end='\r')
            print(fmt_str.format('#' * progress_n, 100), end='', flush=True)
            print()

            num_samples = frame_id
            if num_samples == 0:
                raise ('The number of samples is 0, object path may wrong')

            # Save for later use
            with open(filename_pkl, 'wb') as file:
                if use_seg_data:
                    while num_samples < 8:  # # 样本太少会造成kpconv采样时点数为0
                        print('The number of samples is too small, replication expansion is carried out, current samples number is:{}'.format(num_samples))
                        input_points, input_features, input_labels, num_samples, grasp_obj_name = 2 * input_points, 2 * input_features, 2 * input_labels, 2 * num_samples, 2 * grasp_obj_name
                    print('After replication expansion, the samples number is:{}'.format(num_samples))
                    pickle.dump((input_points, input_features, input_labels, num_samples, grasp_obj_name), file)
                else:
                    pickle.dump((input_points, input_labels, num_samples, grasp_obj_name), file)
                print('save file to ', filename_pkl)

        print('get graspdata information of {} samples in {:.1f}s'.format(num_samples, time.time() - t0))

        if use_seg_data:
            # Specify the serial number of the object for testing as required
            val_idx = np.array([4, 5, 8, 9, 17, 18, 22, 28, 29, 27, 32, 35, 43, 45, 53, 60, 62, 73, 74, 82, 87, 90, 92, 96, 97, 101, 107, 113])
            print(self.mode)
            if self.mode == 'train':
                data_idx = np.setdiff1d(np.arange(len(grasp_obj_name)), val_idx)
                assert len(grasp_obj_name) == 129
            elif self.mode in ['val', 'test']:
                data_idx = val_idx
            else:
                raise ValueError('Wrong set, must be in [training, val, test]')
            print('data_idx is: ', data_idx)

            input_points_, input_features_, input_labels_, grasp_obj_name_ = [], [], [], []
            num_samples = data_idx.shape[0]
            for iidx in range(num_samples):
                input_points_ += [input_points[data_idx[iidx]]]
                input_features_ += [input_features[data_idx[iidx]]]
                input_labels_ += [input_labels[data_idx[iidx]]]
                grasp_obj_name_ += [grasp_obj_name[data_idx[iidx]]]
            input_points, input_features, input_labels, grasp_obj_name = input_points_, input_features_, input_labels_, grasp_obj_name_

        return input_points, input_features, input_labels, num_samples, grasp_obj_name


#####################################################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class GraspNetSampler(Sampler):
    """Sampler for GraspNet"""

    def __init__(self, dataset: GraspNetDataset, use_potential=True, is_arange=False):
        Sampler.__init__(self, dataset)

        # Does the sampler use potential for regular sampling
        self.use_potential = use_potential
        self.is_arange = is_arange

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Create potentials
        if self.use_potential:
            self.potentials = np.random.rand(len(dataset.input_labels)) * 0.1 + 0.1
        else:
            self.potentials = None

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = 10000

        return

    def __iter__(self):
        """
        Yield next batch indices here
        """
        if self.is_arange:
            gen_indices = np.arange(self.dataset.num_samples) * self.dataset.label_max
            print('self.dataset.num_samples is:', self.dataset.num_samples)
        else:
            # # 使用use_potential则每一个batch不出现重复的物体
            if self.use_potential:
                gen0 = np.random.permutation(self.dataset.num_samples)[:self.dataset.epoch_n]
                gen1 = np.random.permutation(self.dataset.num_samples * self.dataset.label_max)[:gen0.shape[0]] % self.dataset.label_max
                gen_indices = gen0 * self.dataset.label_max + gen1
            else:
                gen_indices = np.random.permutation(self.dataset.num_samples * self.dataset.label_max)[:self.dataset.epoch_n]

        ################
        # Generator loop
        ################

        # Initialize concatenation lists
        ti_list = []
        batch_n = 0

        # Generator loop
        for p_ij in gen_indices:
            p_i = p_ij // self.dataset.label_max

            # Size of picked cloud
            n = self.dataset.input_points[p_i].shape[0]

            # In case batch is full, yield it and reset it
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            # Add data to current batch
            ti_list += [p_ij]

            # Update batch size
            batch_n += n

        if len(ti_list)>1:
            yield np.array(ti_list, dtype=np.int32)

        return 0

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return None

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = False

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.grasp_pkl_path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                   self.dataset.config.batch_num)
        if key in batch_lim_dict:
            self.batch_limit = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.grasp_pkl_path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.conv_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.labels)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                       self.dataset.config.batch_num)
            batch_lim_dict[key] = self.batch_limit
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class GraspNetCustomBatch:
    """Custom batch definition with memory pinning for GraspNet"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 5) // 4

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.model_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.model_inds = self.model_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.model_inds = self.model_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def GraspNetCollate(batch_data):
    return GraspNetCustomBatch(batch_data)
