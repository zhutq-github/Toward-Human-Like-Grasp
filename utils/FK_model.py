# -*- coding: UTF-8 -*-
import torch
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

class Barrett_FK(object):
    """
    self.run()为主函数
    输入为base的姿态 (F, 7): x y z qw qx qy qz
         rotations (F, J): J个关节的旋转角度，弧度制
    输出：各个关节的空间坐标（包括base的），与base相同参考系

    使用示例：
    from FK_model import Barrett_FK
    fk = Barrett_FK()
    j_p = fk.run(base, rotations)
    """
    def __init__(self, npy_dir='./utils'):
        self.npy_dir = npy_dir

    def transforms_multiply(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

    def transforms_blank(self, shape0, shape1):
        """
        transforms : (F, J, 4, 4) ndarray
            Array of identity transforms for
            each frame F and joint J
        """
        diagonal = torch.eye(4).to(self.device)
        ts = diagonal.expand(shape0, shape1, 4, 4)
        return ts

    def transforms_base(self, base):
        """
        base：（F, 7）—— x y z qw qx qy qz
        """
        base = base.unsqueeze(-2)
        rotations = base[:, :, 3:]
        q_length = torch.sqrt(torch.sum(rotations.pow(2), dim=-1))  # [F,J,1]
        qw = rotations[..., 0] / q_length  # [F,J,1]
        qx = rotations[..., 1] / q_length  # [F,J,1]
        qy = rotations[..., 2] / q_length  # [F,J,1]
        qz = rotations[..., 3] / q_length  # [F,J,1]
        """Unit quaternion based rotation matrix computation"""
        x2 = qx + qx  # [F,J,1]
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], -1)
        dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], -1)
        dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], -1)
        R = torch.stack([dim0, dim1, dim2], -2).to(self.device)   # [F,1,3,3]

        T = base[..., :3].unsqueeze(-1).to(self.device)  # (F, 1, 3, 1)
        zeros = torch.zeros([int(base.shape[0]), 1, 1, 3]).to(self.device)  # (F, 1, 1, 3)
        ones = torch.ones([int(base.shape[0]), 1, 1, 1]).to(self.device)  # (F, 1, 1, 1)
        base_M = torch.cat([torch.cat([R, zeros], -2), torch.cat([T, ones], -2)], -1) # (F, 1, 4, 4)

        return base_M   # [F,1,4,4]

    def transforms_rotations(self, rotations):
        """
        角度输入暂定为弧度制，最好加上非线性处理，使角度值不超过限制
        rotations : (F, J) , Angle for each frame F and joint J
        M_r: 将角度转为绕z轴的旋转矩阵
        """
        m11 = torch.cos(rotations)  # (F, J)
        m12 = -torch.sin(rotations)  # (F, J)
        m21 = torch.sin(rotations)  # (F, J)
        m22 = torch.cos(rotations)  # (F, J)
        mr = torch.stack([torch.stack([m11, m21], -1), torch.stack([m12, m22], -1)], -1)  # (F, J, 2, 2)
        zeros = torch.zeros([int(rotations.shape[0]), int(rotations.shape[1]), 2, 2]).to(self.device)  # (F, J, 2, 2)
        eyes = torch.eye(2).expand(int(rotations.shape[0]), int(rotations.shape[1]), 2, 2).to(self.device)  # (F, J, 2, 2)
        M_r = torch.cat([torch.cat([mr, zeros], -2), torch.cat([zeros, eyes], -2)], -1)  # (F, J, 4, 4)

        return M_r   # [F,J,4,4]

    def transforms_local(self, M_sh, rotations):
        M_r = self.transforms_rotations(rotations).to(self.device)# [F,J,4,4]
        M_sh = M_sh.expand(int(rotations.shape[0]), int(rotations.shape[1]), 4, 4).to(self.device)
        transforms = self.transforms_multiply(M_sh, M_r)  # [F,J,4,4]
        return transforms

    def transforms_global(self, base, parents, M_sh, rotations, ass_idx=(0,1,2,3)):
        locals = self.transforms_local(M_sh, rotations)  # [F,J,4,4]
        globals = self.transforms_blank(int(rotations.shape[0]), int(rotations.shape[1]))  # [F,J,4,4]
        base_M = self.transforms_base(base)   # [F,1,4,4]

        globals = torch.cat([base_M, globals], 1)  # 0号坐标是基座，直接由网络预测，不参与计算，但是需要给定值 # [F,1+J,4,4]
        globals = torch.split(globals, 1, 1)  # 因为torch.split输出是tuple型，后续无法迭代，所以需要变成list型
        globals = list(globals)  # list长度为J+1，每个元素[F, 1, 4, 4]

        # # Chose ass key joints
        # # 目前，M_ass共2组，来自一种手指的两个link，每组2个
        index = [-1,    -1, -1, 0, 1,    -1, -1, 0, 1,   -1,  0,  1]
        j_num = sum(i > -1 for i in index)
        ass_num = len(ass_idx)
        ass = self.transforms_blank(int(rotations.shape[0]), j_num*ass_num)  # 构建辅助点数组，点数A=j_num×ass_num，总尺寸[F, A, 4, 4]
        ass = torch.split(ass, 1, 1)  # 因为torch.split输出是tuple型，后续无法迭代，所以需要变成list型
        ass = list(ass)  # list长度为A，每个元素[F, 1, 4, 4]
        M_ass = np.load(self.npy_dir+'/M_ass.npy')  # [2,2,4,4] 加载辅助点信息
        # M_ass = np.load(self.npy_dir+'/M_xyz.npy')  # [2,2,4,4] 加载辅助点信息
        M_ass = torch.from_numpy(M_ass).float().to(self.device)
        j_idx = -1

        # Calculate key points
        for i in range(1, len(parents)):  # 从1号而非0号开始，因为0号是基准，已经有了
            # # ↓ 这里实质就是通过不断右乘新矩阵得到本关节相对初始坐标系的变换关系，恰好4×4矩阵最右上角三个数就是本关节在初始坐标系的坐标
            globals[i] = self.transforms_multiply(globals[parents[i]][:, 0], locals[:, i-1])[:, None, :, :]  # 一次右乘（A*M=B）：以A坐标系为基础，进行M的变换，得到B
            if index[i] != -1 and ass_num != 0:
                j_idx = j_idx + 1
                for ass_i in range(ass_num):
                    # #　由于barrett的特殊性，需要辅助点的关节有固定的偏转，本代码无法适应，因此做如下应急方案!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    if index[i] == 0:
                        ass[ass_num * j_idx + ass_i] = self.transforms_multiply(globals[parents[i]][:, 0], M_ass[index[i], ass_idx[ass_i]])[:, None, :, :]
                    else:
                        ass[ass_num*j_idx+ass_i] = self.transforms_multiply(globals[i][:, 0], M_ass[index[i], ass_idx[ass_i]])[:, None, :, :]

        globals = torch.cat(globals, 1)  # [F,1+J,4,4]
        ass = torch.cat(ass, 1)  # [F,A,4,4]
        globals_ass = torch.cat([globals, ass], 1)  # [F,1+J+A,4,4]

        return globals_ass   # [F,1+J+A,4,4]
    '''
    对比说明
                  plam       index           middle         thumb
    key_points  = [ 0,     1,  2, 3, 4,     5,  6, 7, 8,    9, 10, 11]
    parents     = [-1,     0,  1, 2, 3,     0,  5, 6, 7,    0,  9, 10]
    index (ass) = [-1,    -1, -1, 0, 1,    -1, -1, 0, 1,   -1,  0,  1] barrett因关节有固定偏转，所以需要由围绕parents，改为围绕key的
    '''
    def run(self, base, rotations, ass_idx=(0,1,2,3)):  # base:[F,1,3], rotations:[F,J,4]
        self.device = base.device
        # # ↓ 28个关键点所对应的运动链父结点
        parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10]   # 依据grand
        M_sh = np.load(self.npy_dir+'/M_barrett.npy')  # 加载barrett的运动学关系
        M_sh = torch.from_numpy(M_sh).float()
        # print(M_sh.shape)
        positions = self.transforms_global(base, parents, M_sh, rotations, ass_idx)[:, :, :, 3]  # [F,1+J+A,1,4] --> [F,1+J+A,4]
        return positions[:, :, :3]  # positions[:, :, :3] / positions[:, :, 3, None]   # [F,1+J+A,3]

# # ↓ for test
def fk_run(outputs_r, outputs_t, outputs_a, ass_idx=(0,1)):
    # # 输入正向运动学层
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
    outputs_FK = fk.run(outputs_base, outputs_rotation * 1.5708, ass_idx)  # [F, 1+J+A, 3]  #原始1+J个关键点，再加上A个辅助点

    return outputs_FK