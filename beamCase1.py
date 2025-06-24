import numpy as np
from matplotlib import pyplot as plt
import mesh as msh
import plotting

print('==================================================')
print('Beam Test Case 1: Linear Distributed Load Full Beam')
print('==================================================')
n_ed = int(input('Please enter degrees of freedom (only 2 is supported currently): ')) # degrees of freedom per node
l = int(input('Please enter beam length in meters: ')) # Length of beam
h = int(input('Please enter beam height in meters: ')) # Width of beam
if n_ed == 3:
    w = int(input('Please enter beam width in meters: ')) # Width of beam
xnum = int(input('Please enter number of x nodes: ')) # Number of horizontal nodes)
ynum = int(input('Please enter number of y nodes: ')) # Number of vertical nodes
if n_ed == 3:
    znum = int(input('Please enter number of z nodes: ')) # Number of horizontal nodes)

# store number of total nodes
n_n = xnum * ynum

# Create 2 nparrays which set up the x and y points for meshgrid
xnodes = np.linspace(0, l, xnum)
ynodes = np.linspace(0, h, ynum)

# Initialie Nodes list to whole objects of class Node
nodes = np.zeros(n_n, dtype=object)


# for now everything is only 2D
if n_ed == 2:
    # nodes_List takes in the empty nodes list, the degree of freedom of the node
    # and the x and y lists of locations and returns an np array of objects of type Node
    nodes = msh.nodes_List(nodes, n_ed, xnodes, ynodes)

    # Plot Node Mesh
    plotting.nodes_Plot(nodes, l, h)


    # BQE returns an nparray of Bilinear Quadrilateral Elements with Beam conventional ordering
    # need to input xnum and ynum for correct element node pairing
    elements = msh.BQE(nodes, xnum, ynum)

    # Plot Node and Element Mesh
    plotting.mesh_Plot(nodes, elements, l, h)



