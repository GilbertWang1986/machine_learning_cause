from __future__ import print_function
from __future__ import division

import cPickle

import numpy as np
import matplotlib.pylab as plt



R = np.linspace(2, 10, 2000)
E = np.power(2.2/R, 12.0) - np.power(2.2/R, 6.0) 

cPickle.dump(np.hstack((R.reshape(-1,1), E.reshape(-1,1))), open('reg','w'))

# plt.plot(R, E, 'bo')
# plt.xlabel('distance')
# plt.ylabel('energy')
# plt.savefig('regression.pdf')
# plt.show()

