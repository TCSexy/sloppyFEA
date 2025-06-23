import numpy as np
from matplotlib import pyplot as plt
from ..meshing import mesh



print('==================================================')
print('Beam Test Case 1: Linear Distributed Load Full Beam')
print('==================================================')
l = int(input('Please enter beam length in meters: ')) # Length of beam
w = int(input('Please enter beam width in meters: ')) # Width of beam
xnodes = int(input('Please enter number of x nodes: ')) # Number of horizontal nodes)
ynodes = int(input('Please enter number of y nodes: ')) # Number of vertical nodes


mesh.beamMesh(l, w, xnodes, ynodes)
print('1')


