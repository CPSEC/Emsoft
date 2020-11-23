from control.matlab import ss, lsim, linspace, c2d
from functools import partial
from state_estimation import Estimator
import numpy as np
import math


class Exp:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo']
    count = 0

    def __init__(self, sysc, Ts, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 150, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None


class vt:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc']
    count = 0
    A = [[-25 / 3]]
    B = [[5]]
    C = [[1]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.02

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 0.5
        self.i = 7
        self.d = 0
        self.total = 10
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref = [0] * 251 + [1] * 250
        self.thres = 5
        self.drift = 0.12
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 350
        self.maxc = 5
        # --------------------------------------
        self.safeset = {'lo': [-2.7], 'up': [2.7]}


class dc:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure']
    count = 0
    J = 0.01
    b = 0.1
    K = 0.01
    R = 1
    L = 0.5

    A = [[0, 1, 0],
         [0, -b / J, K / J],
         [0, -K / L, -R / L]]
    B = [[0], [0], [1 / L]]
    C = [[1, 0, 0]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.2

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 11
        self.i = 0
        self.d = 5
        self.total = 24
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref = [math.pi / 2] * 71 + [-math.pi / 2] * 50
        self.thres = 0.2
        self.drift = 0.01
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 20
        self.maxc = 20
        self.xmeasure = [[0], [0], [0]]
        self.xreal = [[0], [0], [0]]
        self.safeset = {'lo': [-4, -1000, -1000], 'up': [4, 1000, 1000]}


class ap:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure']
    A = [[-0.313, 56.7, 0],
         [-0.0139, -0.426, 0],
         [0, 56.7, 0]]
    B = [[0.232], [0.0203], [0]]
    C = [[0, 0, 1]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.02

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 14
        self.i = 0.8
        self.d = 5.7
        self.total = 10
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref = [0.5] * 201 + [0.7] * 200 + [0.5] * 100
        self.thres = 10
        self.drift = 1
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 300
        self.maxc = 20
        self.xmeasure = [[0], [0], [0]]
        self.xreal = [[0], [0], [0]]
class qd:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure']
    count = 0
    g = 9.81
    m = 0.468
    A = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
    B = [[0], [0], [0], [0], [0], [0], [0], [0], [1 / m], [0], [0], [0]]
    C = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    D = [[0], [0], [0], [0], [0], [0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.02

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 0.1
        self.i = 0
        self.d = 0.6
        self.total = 30
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref =  [2] * 601 + [4] * 600 + [2] * 300
        self.thres = 3
        self.drift = 1
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 700
        self.maxc = 50
        self.xmeasure = [[0], [0], [0],[0], [0], [0],[0], [0], [0]]
        self.xreal = [[0], [0], [0],[0], [0], [0],[0], [0], [0]]
        self.safeset = {'lo': [-100] * 8 + [-1], 'up': [100] * 8 + [8]}