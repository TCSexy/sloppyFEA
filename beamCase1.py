import numpy as np
from matplotlib import pyplot as plt
import mesh as msh
import plotting
import feasolve as solve



#======================================================================
# INTERFACE 
#======================================================================

print('==================================================')
print('Beam Test Case 1: Linear Distributed Load Full Beam')
print('==================================================')
n_ed = int(input('Please enter degrees of freedom (only 2 is supported currently): ')) # degrees of freedom per node
l = float(input('Please enter beam length in meters: ')) # Length of beam
h = float(input('Please enter beam height in meters: ')) # Height of beam
t = float(input('Please enter beam thickness in meters: ')) # Width of beam
xnum = int(input('Please enter number of x nodes: ')) # Number of horizontal nodes)
ynum = int(input('Please enter number of y nodes: ')) # Number of vertical nodes
if n_ed == 3:
    znum = int(input('Please enter number of z nodes: ')) # Number of horizontal nodes)

#======================================================================
# MESH
#======================================================================

# store number of total nodes
n_n = xnum * ynum

# Create 2 nparrays which set up the x and y points for meshgrid
xnodes = np.linspace(0, l, xnum)
ynodes = np.linspace(0, h, ynum)

# Initialie Nodes list to whole objects of class Node
nodes = np.zeros(n_n, dtype=object)


# for now everything is only 2D


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

ID, IEN = msh.generate_ID_IEN(nodes, elements, n_ed)

#======================================================================
# BOUNDARY CONDITIONS
#======================================================================

# Now it is time to add boundary conditions
# Should also return number of global equations
ID, n_eq = msh.apply_boundary_conditions(nodes, ID)


#======================================================================
# MATRIX ASSEMBLY
#======================================================================

# Setup the linear equation. Need stiffness matrix and force vector
# solution_setup takes in the ID and IEN arrays, the list of elements
# and the list of nodes and returns the global stiffness matrix and 
# the force vector
P = float(input('Enter load in kN (+ for up, - for down): '))
p = P*1000
traction = np.array([0, p])
K, F = solve.solution_setup(ID, IEN, elements, n_eq, h, t, traction)

#======================================================================
# Solve
#======================================================================

displacements = np.linalg.solve(K, F)


#======================================================================
# Post Processing
#======================================================================

# Assign displacements to each node
for j, node in enumerate(nodes):
    node_disps = np.zeros(n_ed)
    for i in range(n_ed):
        global_dof = ID[i, j] # Global dof number for node j, dof i
        if global_dof != 0:
            node_disps[i] = displacements[global_dof - 1]
        else: # Fixed dof
            node_disps[i] = 0.0
    node.add_displacement(node_disps)

plotting.deformed_plot(nodes, l, h, xnum, ynum) 

# Lets compare displacements to theory
# I am going to apply different thicknesses in the I calculation
# to determine which one is accurate

E = 70e9
I = (t*h**3)/12
c = traction[1]
theory = lambda x: (1/(E*I))*((c*x**4)/24 - (c*l*x**3)/6 + (c*l*l*x**2)/4)

# Call the comparison function after deformed_plot
plotting.compare_theory(nodes, xnum, l, t, theory)