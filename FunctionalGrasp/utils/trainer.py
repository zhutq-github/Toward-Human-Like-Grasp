# Basic libs
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys

from utils.config import Config

from utils.write_xml import write_xml
from utils.FK_model import Barrett_FK

heloss = nn.HingeEmbeddingLoss()
l2loss = nn.MSELoss()

class ModelTrainer:

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.Adam([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate)
                                         # momentum=config.momentum,
                                         # weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Function
    def subset_dis(self, outputs_FK, idx1, idx2):
        subset_dis = ((outputs_FK[:, idx1].unsqueeze(2).expand(-1, -1, outputs_FK[:, idx2].shape[1], -1)
                     - outputs_FK[:, idx2].unsqueeze(1).expand(-1, outputs_FK[:, idx1].shape[1], -1, -1)) ** 2).sum(-1).sqrt().reshape(outputs_FK.shape[0], -1)
        return subset_dis
    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config, debug=False):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps                 loss                   train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)

            # xml directory
            results_xml_path = join(config.saving_path, 'xml')
            if not exists(results_xml_path):
                # os.rename(results_path, results_path[:-2] + 'aaa-' + str(time.strftime("%Y-%m-%d-%H-%M-%S")))
                makedirs(results_xml_path + '/train')
                makedirs(results_xml_path + '/val')

        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        for epoch in range(config.max_epoch):

            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0
            for ib, batch in enumerate(training_loader):

                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs_r, outputs_t, outputs_a = net(batch, config)
                # outputs = torch.cat((outputs_r, outputs_t, outputs_a), 1)

                # # 角度loss
                angle_lower = torch.tensor([  0.,   0.,   0.,   0.]).cuda() / 90.0  # * 1.5708 / 1.5708
                angle_upper = torch.tensor([30., 90., 90., 90.]).cuda() / 90.0  # In order to ensure a better hand shape, the reasonable range is narrowed
                angle_lower_pair = torch.zeros([2, outputs_a.reshape(-1).shape[0]]).cuda()
                angle_upper_pair = torch.zeros([2, outputs_a.reshape(-1).shape[0]]).cuda()
                angle_lower_pair[0] = angle_lower.repeat(outputs_a.shape[0]) - outputs_a.reshape(-1)
                angle_upper_pair[0] = outputs_a.reshape(-1) - angle_upper.repeat(outputs_a.shape[0])
                loss_angles = (torch.max(angle_lower_pair, 0)[0] + torch.max(angle_upper_pair, 0)[0]).sum()

                # # 四元数 weight decay
                loss_rotate = (outputs_r ** 2).sum() / (2 * outputs_r.shape[0])

                # # 输入正向运动学层
                # 正则化
                outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)
                # 3 + 4
                outputs_base = torch.cat((outputs_t / 5.0 * 1000, outputs_r), 1)
                # 4 -> 11
                outputs_rotation = torch.zeros([outputs_a.shape[0], 11]).type_as(outputs_a)  # .cuda()
                outputs_rotation[:, 0] = outputs_a[:, 0]
                outputs_rotation[:, 1] = outputs_a[:, 1]
                outputs_rotation[:, 2] = 0.333333333 * outputs_a[:, 1]
                outputs_rotation[:, 4] = outputs_a[:, 0]
                outputs_rotation[:, 5] = outputs_a[:, 2]
                outputs_rotation[:, 6] = 0.333333333 * outputs_a[:, 2]
                outputs_rotation[:, 8] = outputs_a[:, 3]
                outputs_rotation[:, 9] = 0.333333333 * outputs_a[:, 3]
                fk = Barrett_FK()
                ass_idx = (0, 1)  # using inner & back keypoint
                outputs_FK = fk.run(outputs_base, outputs_rotation * 1.5708, ass_idx)

                # # 手自碰撞约束loss_handself
                hand_self_distance0 = self.subset_dis(outputs_FK, [4], [8, 11])/30
                hand_self_distance1 = self.subset_dis(outputs_FK, [8], [11])/30
                # 求和
                hand_self_distance = torch.cat([hand_self_distance0, hand_self_distance1], 1).reshape(-1)
                hand_self_pair = torch.zeros([2, hand_self_distance.shape[0]]).cuda()
                hand_self_pair[0] = 1 - hand_self_distance
                loss_handself = torch.max(hand_self_pair, 0)[0].sum() / outputs_FK.shape[0]

                # # 接近和远离约束: loss_close / loss_away
                # ↓ For close and away. Select the serial number of the required touch code.
                fetures_close_mask = [0, 3, 6]  # Simplified version that uses only three fingertips to touch objects.
                fetures_away_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15]

                batch_points = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 3).cuda()  # [F, 20000, 3] 为了批量计算，以batch中点数最多为初始化
                batch_28 = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), outputs_FK.shape[-2]).cuda()
                batch_features_close = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), len(fetures_close_mask)).cuda()
                batch_features_away = torch.ones(batch.model_inds.shape[0], int(max(batch.lengths[0])), len(fetures_away_mask)).cuda()

                i_begin = 0
                for pi in range(batch.model_inds.shape[0]):
                    batch_points[pi, :batch.lengths[0][pi]] = batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000
                    batch_28[pi, :batch.lengths[0][pi]] = 1
                    batch_features_close[pi, :batch.lengths[0][pi]] = batch.features[i_begin:i_begin + batch.lengths[0][pi], fetures_close_mask]
                    batch_features_away[pi, :batch.lengths[0][pi]] = batch.features[i_begin:i_begin + batch.lengths[0][pi], fetures_away_mask]
                    batch_features_away[pi] = (batch_features_away[pi] - 1) ** 2
                    i_begin = i_begin + batch.lengths[0][pi]

                batch_distance = ((batch_points.unsqueeze(2).expand(-1,-1,outputs_FK.shape[1],-1) - outputs_FK.unsqueeze(1).expand(-1,batch_points.shape[1],-1,-1))**2).sum(-1).sqrt()  # [F, 20000, ?]

                # Heuristic penetration loss
                batch_dis = batch_distance * batch_28
                batch_dis[batch_dis == 0] = float("inf")
                batch_dis_idx = batch_dis.argmin(dim=1)
                batch_dis = batch_points[np.arange(batch_dis_idx.shape[0]).repeat(batch_dis_idx.shape[1]), batch_dis_idx.reshape(-1)]
                loss_out = (batch_dis ** 2).sum(-1).sqrt() - ((outputs_FK.reshape(-1, 3)) ** 2).sum(-1).sqrt()
                loss_out[loss_out <= 0] = 0
                loss_out = loss_out.sum() / outputs_FK.shape[0]

                idx_close = (np.array([22,14,18])).tolist()  # Specify the required keypoints
                batch_dis_close = batch_distance[:, :, idx_close] * batch_features_close
                batch_dis_close[batch_features_close == 0] = float("inf")
                weight_close = [10, 8, 6]
                loss_close = torch.min(batch_dis_close, -2)[0] * torch.tensor(weight_close).cuda()
                loss_close[loss_close == float("inf")] = 0
                loss_close = loss_close.sum() / batch_dis_close.shape[0]

                '''
                对比说明
                              plam       index           middle         thumb
                key_points  = [ 0,     1, 2, 3, 4,     5, 6, 7, 8,    9, 10, 11]
                parents     = [-1,     0, 1, 2, 3,     0, 5, 6, 7,    0,  9, 10]
                index (ass) = [-1,    -1, 0, 1, 2,    -1, 0, 1, 2,    0,  1,  2]
                '''
                idx_away = [11, 10, 9, 4, 3, 2, 8, 7, 6, 0]  # Specify the required keypoints
                batch_dis_away = batch_distance[:, :, idx_away] * batch_features_away
                batch_dis_away[batch_features_away == 0] = float("inf")
                threshold_away = [10, 15, 20, 10, 15, 20, 10, 15, 20, 50]
                loss_away = torch.log2((torch.tensor(threshold_away).cuda() + 5) / (torch.min(batch_dis_away, -2)[0] + 0.01))
                weight_away = [1, 1, 2, 1, 1, 2, 1, 1, 2, 5]
                loss_away = torch.tensor(weight_away).cuda() * loss_away
                loss_away[loss_away <= 0] = 0
                loss_away = loss_away.sum() / batch_dis_away.shape[0]

                idx_always_away = [13,15,17,19,21,23]  # Specify the required keypoints
                batch_dis_away = batch_distance[:, :, idx_always_away]
                threshold_away = [30, 16] * 3
                loss_always_away = torch.log2((torch.tensor(threshold_away).cuda()) / (torch.min(batch_dis_away, -2)[0] + 0.01))
                weight_away = [1, 2] * 3
                loss_always_away = torch.tensor(weight_away).cuda() * loss_always_away
                loss_always_away[loss_always_away <= 0] = 0
                loss_always_away = loss_always_away.sum() / batch_dis_away.shape[0]

                if self.epoch % 100 in [45]:
                    loss = 10 * loss_handself + 100 * loss_angles + 0.5*loss_rotate + loss_out + loss_always_away
                elif self.epoch % 100 in [95]:
                    loss = 10 * loss_handself + 0.4 * loss_close + 1.0 * loss_away + 100 * loss_angles + 0.5*loss_rotate + loss_out + loss_always_away
                else:
                    loss = 10 * loss_handself + 0.6 * loss_close + 2 * (loss_away + loss_always_away) + 10 * loss_angles + 0.5*loss_rotate


                acc = loss
                t += [time.time()]

                # Backward + optimize
                loss.backward()


                if config.grad_clip_norm > 0:
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L=ang:{:.2f} + self:{:.2f} + out:{:.2f} + (r{:.2f}) + close:{:.2f} + away:{:.2f}_ala{:.2f}={:.2f} acc={:3.1f}% / lr={} / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss_angles.item(), loss_handself.item(), loss_out.item(), loss_rotate.item(),
                                         loss_close.item(), loss_away.item(), loss_always_away.item(), loss.item(),
                                         100 * acc, self.optimizer.param_groups[0]['lr'],
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = 'e{:03d}-i{:04d} => ang:{:.2f} + self:{:.2f} + out:{:.2f} + (r{:.2f}) + close:{:.2f} + away:{:.2f}_ala{:.2f}, acc:{:.3f}%, time:{:.3f}\n'
                        file.write(message.format(self.epoch, self.step,
                                                  loss_angles, loss_handself, loss_out, loss_rotate,
                                                  loss_close, loss_away, loss_always_away,  # net.reg_loss,
                                                  acc * 100.0,
                                                  t[-1] - t0))

                self.step += 1

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]
                    # print('11111111111111-=-=-=-=-=-222222222222222222-=-=-=-=-=-=333333333333333333 : ', param_group['lr'])

            if self.epoch % 10 == 9:
                grip_pkl = join(results_xml_path, 'train/grip_save_{:04d}.pkl'.format(self.epoch))
                grip_r = outputs_r.clone().detach().cpu().numpy()
                grip_t = outputs_t.clone().detach().cpu().numpy() / 5.0 * 1000
                grip_a = outputs_a.clone().detach().cpu().numpy() * 1.5708
                grip_idx = batch.model_inds.clone().detach().cpu().numpy()
                obj_name = training_loader.dataset.grasp_obj_name
                print('1111111111111111111111111111111111111', grip_r.shape,  grip_t.shape, grip_a.shape, grip_idx.shape, len(obj_name))
                with open(grip_pkl, 'wb') as file:
                    pickle.dump((grip_r, grip_t, grip_a, grip_idx, obj_name), file)
                    print('save file to ', grip_pkl)

                for i in range(grip_r.shape[0]):
                    write_xml(obj_name[grip_idx[i]], grip_r[i], grip_t[i], grip_a[i], path=results_xml_path + '/train/epoch{:04d}_{}_{}.xml'.format(self.epoch, grip_idx[i], i), mode='train')

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path, _use_new_zipfile_serialization=False)

                # Save checkpoints occasionally
                if (self.epoch) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch))
                    torch.save(save_dict, checkpoint_path, _use_new_zipfile_serialization=False)

            # Validation
            net.eval()
            if self.epoch % 10 == 9:
                self.validation(net, val_loader, config, results_xml_path, epoch=self.epoch-1)
            net.train()

        print('Finished Training')
        return

    # Validation methods ，目前的验证方法是和稳定抓取的标签做对比，即预训练的验证，后续开源代码应该去掉
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config, results_path, epoch=None):

        if config.dataset_task == 'regression':
            self.object_grasping_validation(net, val_loader, config, results_path, epoch)
        else:
            raise ValueError('No validation method implemented for this network type')

    def object_grasping_validation(self, net, val_loader, config, results_xml_path, epoch):
        """
        Perform a round of validation and show/save results
        :param net: network object
        :param val_loader: data loader for validation set
        :param config: configuration object
        """

        #####################
        # Network predictions
        #####################

        val_grip_r = []
        val_grip_t = []
        val_grip_a = []
        val_grip_idx = []
        val_obj_name = val_loader.dataset.grasp_obj_name

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in val_loader:

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs_r, outputs_t, outputs_a = net(batch, config)
            outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)

            # Get probs and labels
            val_grip_r += [outputs_r.cpu().detach().numpy()]
            val_grip_t += [outputs_t.cpu().detach().numpy() / 5.0 * 1000]
            val_grip_a += [outputs_a.cpu().detach().numpy() * 1.5708]
            val_grip_idx += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * len(val_grip_idx) / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        val_grip_r = np.vstack(val_grip_r)
        val_grip_t = np.vstack(val_grip_t)
        val_grip_a = np.vstack(val_grip_a)
        val_grip_idx = np.hstack(val_grip_idx)

        #####################
        # Save predictions
        #####################
        grip_path = join(results_xml_path, 'val', 'epoch_'+('{:04d}'.format(epoch)))
        if not os.path.exists(grip_path):
            os.makedirs(grip_path)
        grip_pkl = join(grip_path, 'grip_save_{:04d}_val.pkl'.format(epoch))
        print('1111111111111111111111111111111111111', val_grip_r.shape, val_grip_t.shape, val_grip_a.shape, val_grip_idx.shape, len(val_obj_name))
        with open(grip_pkl, 'wb') as file:
            pickle.dump((val_grip_r, val_grip_t, val_grip_a, val_grip_idx, val_obj_name), file)
            print('save file to ', grip_pkl)

        for i in range(val_grip_r.shape[0]):
            write_xml(val_obj_name[val_grip_idx[i]], val_grip_r[i], val_grip_t[i], val_grip_a[i], path=join(grip_path, 'epoch{:04d}_{}_{}_val.xml'.format(epoch, val_grip_idx[i], i)), mode='val')

        return None
