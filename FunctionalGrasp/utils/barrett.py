'''
D-H parameter: https://blog.csdn.net/xiaolongwoaini99/article/details/80391462
'''
import numpy as np
import math

def quat2mat(q):
    R = np.array([1-2*(q[1]*q[1]+q[2]*q[2]), 2*(q[0]*q[1]-q[2]*q[3]), 2*(q[0]*q[2]+q[1]*q[3])],
                 [2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[0]*q[0]+q[2]*q[2]), 2*(q[1]*q[2]-q[0]*q[3])],
                 [2*(q[0]*q[2]-q[1]*q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[0]*q[0]+q[1]*q[1])])
    return R

def quat_multi(q1, q2):
    q3 = np.array([0, 0, 0, 0])
    q3[0]=q1[3]*q2[0]+q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]
    q3[1]=q1[3]*q2[1]-q1[0]*q2[2]+q1[1]*q2[3]+q1[2]*q2[0]
    q3[2]=q1[3]*q2[2]+q1[0]*q2[1]-q1[1]*q2[0]+q1[2]*q2[3]
    q3[3]=q1[3]*q2[3]-q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]
    q3=q3/q3.std()
    return q3

def mat4x4_xyz(x, y, z):
    trans = np.mat([[1, 0, 0, x],[0, 1, 0, y],[0, 0, 1, z],[0, 0, 0, 1]])
    return trans

def mat4x4_theta_x(theta_x):
    theta_x = theta_x*np.pi/180
    trans = np.mat([[1,                 0,                  0, 0],
                    [0, math.cos(theta_x), -math.sin(theta_x), 0],
                    [0, math.sin(theta_x),  math.cos(theta_x), 0],
                    [0,                 0,                  0, 1]])
    return trans

def mat4x4_theta_y(theta_y):
    theta_y = theta_y * np.pi / 180
    trans = np.mat([[ math.cos(theta_y), 0, math.sin(theta_y), 0],
                    [                 0, 1,                 0, 0],
                    [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                    [                 0, 0,                 0, 1]])
    return trans

def mat4x4_theta_z(theta_z):
    theta_z = theta_z * np.pi / 180
    trans = np.mat([[math.cos(theta_z), -math.sin(theta_z), 0, 0],
                    [math.sin(theta_z),  math.cos(theta_z), 0, 0],
                    [                0,                  0, 1, 0],
                    [                0,                  0, 0, 1]])
    return trans

p0 = np.mat([0,0,0,1]).T
px = np.mat([10,0,0,1]).T
py = np.mat([0,10,0,1]).T
pz = np.mat([0,0,10,1]).T
p00 = np.hstack((p0, px, py, pz))
# p01 = p00+10
# p02 = p00+20
# p03 = p00+5
# print(p00)
# print(p01)

# # index finger
mi10 = mat4x4_xyz(25, 0, -1)
mi11 = mat4x4_theta_y(180)
MI1 = mi10*mi11

mi20 = mat4x4_theta_z(90)
mi21 = mat4x4_xyz(50, 0, 0)
mi22 = mat4x4_theta_x(-90)
MI2 = mi20*mi21*mi22

mi30 = mat4x4_theta_z(5)
mi31 = mat4x4_xyz(70, 0, 0)
MI3 = mi30*mi31

mi40 = mat4x4_theta_z(40)
mi41 = mat4x4_xyz(55, 0, 0)
MI4 = mi40*mi41

pi1 = np.hstack((MI1*p0, MI1*px, MI1*py, MI1*pz))
pi2 = np.hstack((MI1*MI2*p0, MI1*MI2*px, MI1*MI2*py, MI1*MI2*pz))
pi3 = np.hstack((MI1*MI2*MI3*p0, MI1*MI2*MI3*px, MI1*MI2*MI3*py, MI1*MI2*MI3*pz))
pi4 = np.hstack((MI1*MI2*MI3*MI4*p0, MI1*MI2*MI3*MI4*px, MI1*MI2*MI3*MI4*py, MI1*MI2*MI3*MI4*pz))

# # middle finger
mm10 = mat4x4_xyz(-25, 0, -1)
MM1 = mm10

mm20 = mat4x4_theta_z(90)
mm21 = mat4x4_xyz(50, 0, 0)
mm22 = mat4x4_theta_x(90)
MM2 = mm20*mm21*mm22

mm30 = mat4x4_theta_z(5)
mm31 = mat4x4_xyz(70, 0, 0)
MM3 = mm30*mm31

mm40 = mat4x4_theta_z(40)
mm41 = mat4x4_xyz(55, 0, 0)
MM4 = mm40*mm41

pm1 = np.hstack((MM1*p0, MM1*px, MM1*py, MM1*pz))
pm2 = np.hstack((MM1*MM2*p0, MM1*MM2*px, MM1*MM2*py, MM1*MM2*pz))
pm3 = np.hstack((MM1*MM2*MM3*p0, MM1*MM2*MM3*px, MM1*MM2*MM3*py, MM1*MM2*MM3*pz))
pm4 = np.hstack((MM1*MM2*MM3*MM4*p0, MM1*MM2*MM3*MM4*px, MM1*MM2*MM3*MM4*py, MM1*MM2*MM3*MM4*pz))

# # thumb
mt10 = mat4x4_xyz(0, -50, -1)
mt11 = mat4x4_theta_y(-90)
mt12 = mat4x4_theta_z(-90)
MT1 = mt10*mt11*mt12

mt20 = mat4x4_theta_z(5)
mt21 = mat4x4_xyz(70, 0, 0)
MT2 = mt20*mt21

mt30 = mat4x4_theta_z(40)
mt31 = mat4x4_xyz(55, 0, 0)
MT3 = mt30*mt31

pt1 = np.hstack((MT1*p0, MT1*px, MT1*py, MT1*pz))
pt2 = np.hstack((MT1*MT2*p0, MT1*MT2*px, MT1*MT2*py, MT1*MT2*pz))
pt3 = np.hstack((MT1*MT2*MT3*p0, MT1*MT2*MT3*px, MT1*MT2*MT3*py, MT1*MT2*MT3*pz))

# # save
M_barrett = np.asarray([MI1, MI2, MI3, MI4, MM1, MM2, MM3, MM4, MT1, MT2, MT3])
print(M_barrett.shape)
# np.save('./M_barrett.npy', M_barrett)
M_sh = np.load('../utils/M_barrett.npy')
print((M_barrett == M_sh).all())

# # show
is_show = False
if is_show:
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    import sys

    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 500
    w.show()
    w.setWindowTitle('show Grasping by pyqtgraph')

    # #　制造一些变换
    r90 = mat4x4_theta_z(90)
    MI2 = MI2 * r90
    pi1 = np.hstack((MI1 * p0, MI1 * px, MI1 * py, MI1 * pz))
    pi2 = np.hstack((MI1 * MI2 * p0, MI1 * MI2 * px, MI1 * MI2 * py, MI1 * MI2 * pz))
    pi3 = np.hstack((MI1 * MI2 * MI3 * p0, MI1 * MI2 * MI3 * px, MI1 * MI2 * MI3 * py, MI1 * MI2 * MI3 * pz))
    pi4 = np.hstack((MI1 * MI2 * MI3 * MI4 * p0, MI1 * MI2 * MI3 * MI4 * px, MI1 * MI2 * MI3 * MI4 * py, MI1 * MI2 * MI3 * MI4 * pz))

    # #　正经开始画图
    Joints = np.hstack([p00[:3,0], pi1[:3,0], pi2[:3,0], pi3[:3,0], pi4[:3,0], pm1[:3,0], pm2[:3,0], pm3[:3,0], pm4[:3,0], pt1[:3,0], pt2[:3,0], pt3[:3,0]])
    Joints_T = np.asarray(Joints.T)
    sp1 = gl.GLScatterPlotItem(pos=Joints_T, size=2.5, color=(1, 1, 1, 1), pxMode=False)
    w.addItem(sp1)
    lines = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11]]
    # lines_col = [(0.1, 0.2, 0.4, 1), (0.6, 0.2, 0.7, 1), (0.2, 0.5, 0.6, 1), (1, 0.6, 0, 1),
    #              (0.1, 0.2, 0.4, 1), (0.6, 0.2, 0.7, 1), (0.2, 0.5, 0.6, 1), (1, 0.6, 0, 1),
    #              (0.1, 0.2, 0.4, 1),                     (0.2, 0.5, 0.6, 1), (1, 0.6, 0, 1),
    #             ]
    lines_col = [(0.6, 0.2, 0.7, 1), (0.6, 0.2, 0.7, 1), (0.6, 0.2, 0.7, 1), (0.6, 0.2, 0.7, 1),
                 (0.2, 0.5, 0.6, 1), (0.2, 0.5, 0.6, 1), (0.2, 0.5, 0.6, 1), (0.2, 0.5, 0.6, 1),
                 (1, 0.6, 0, 1),                         (1, 0.6, 0, 1),     (1, 0.6, 0, 1),
                ]
    # lines_col = [(1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1),
    #              (0, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1),
    #              (0, 0, 1, 1),               (0, 0, 1, 1), (0, 0, 1, 1),
    #             ]
    for ii, line in enumerate(lines):
        plt = gl.GLLinePlotItem(pos=Joints_T[line], color=lines_col[ii], width=2, antialias=True)
        w.addItem(plt)

    p_list = [p00*10, pi1, pi2, pi3, pi4, pm1, pm2, pm3, pm4, pt1, pt2, pt3]
    for pl in p_list:
        pl = np.asarray(pl.T)[:, :3]
        plt = gl.GLLinePlotItem(pos=pl[[0,1]], color=(1,0,0,1), width=4, antialias=True)
        w.addItem(plt)
        plt = gl.GLLinePlotItem(pos=pl[[0,2]], color=(0,1,0,1), width=4, antialias=True)
        w.addItem(plt)
        plt = gl.GLLinePlotItem(pos=pl[[0,3]], color=(0,0,1,1), width=4, antialias=True)
        w.addItem(plt)


    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

###################################################### -辅助点一代- ################################################################################
''''''
# # 计算辅助点，尺寸对照说明书，各参数详见参考onenote图
M_F_Middle_0 = mat4x4_xyz(35, 8, 0)
_M_F_Middle_0 = mat4x4_xyz(35, -8, 0)

M_F_Distal_0 = mat4x4_xyz(0, 8, 0)
_M_F_Distal_0 = mat4x4_xyz(0, -8, 0)

M_ass = np.asarray([[M_F_Middle_0,    _M_F_Middle_0],
                    [M_F_Distal_0,    _M_F_Distal_0]])
print(M_ass.shape)
# np.save('./M_ass.npy', M_ass)
M_asss = np.load('../utils/M_ass.npy')
print((M_ass == M_asss).all())

########################################################## -xyz轴- ################################################################################
''''''
# # 计算辅助点，尺寸对照说明书，各参数详见参考onenote图
# M_F_Middle_0 = mat4x4_xyz(-35, 8, 0)
# _M_F_Middle_0 = mat4x4_xyz(-35, -8, 0)
xx = mat4x4_xyz(10, 0, 0)
yy = mat4x4_xyz(0, 10, 0)
zz = mat4x4_xyz(0, 0, 10)

M_ass = np.asarray([[xx, yy, zz]])
print(M_ass.shape)
# np.save('./M_xyz.npy', M_ass)
M_asss = np.load('../utils/M_xyz.npy')
print((M_ass == M_asss).all())
