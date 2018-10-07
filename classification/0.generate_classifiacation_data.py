from __future__ import print_function
from __future__ import division

import cPickle

import numpy as np
import matplotlib.pylab as plt


def generate_data_in_circle(N, point, radius):
    R = np.random.random(N)*radius
    angle = np.random.random(N)*2.0*np.pi
    x = point[0] + R*np.cos(angle)
    y = point[1] + R*np.sin(angle)

    return np.hstack( (x.reshape(-1, 1), y.reshape(-1, 1)) )
    


pts0 = generate_data_in_circle(100, [10,10], 8)
pts1 = generate_data_in_circle(100, [20,20], 8)

pts_A  = pts0[:50, :]
pts_B  = pts1[:50, :]
pts_UA = pts0[50:, :]
pts_UB = pts1[50:, :]

# cPickle.dump(pts_A, open('train_A', 'w'))
# cPickle.dump(pts_B, open('train_B', 'w'))
# cPickle.dump(np.vstack((pts_UA,pts_UB)), open('pre', 'w'))


# plt.plot(pts_A[:, 0], pts_A[:, 1], 'ro')
# plt.plot(pts_B[:, 0], pts_B[:, 1], 'bo')
# plt.plot(pts_UA[:, 0], pts_UA[:, 1], 'mo')
# plt.plot(pts_UB[:, 0], pts_UB[:, 1], 'mo')

# plt.xlim([0, 40])
# plt.ylim([0, 40])
# plt.legend(['Class A', 'Class B', 'Unknown'])
# plt.savefig('classification.pdf')
# plt.show()