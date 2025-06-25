import numpy as np

# A node stores the degree of freedom of the node, and 
# the (x,y) or (x,y,z) point in an nparray
class Node:

    # n_ed is an integer which represents the degree of freedom per node
    # point is an nparray with the coordinates of the node
    # global_num is the global node number
    def __init__(self, n_ed, point, global_num):
        self.n_ed = n_ed
        self.point = point
        self.global_num = global_num


# Each element contains an nparray of all the nodes per element
# Each element contains a global numbering
class Element:
    
    # nodes is an ordered nparray of nodes within the element
    # global_num is the global element number
    def __init__(self, nodes, global_num):
        self.nodes = nodes # the local numbering of nodes is inherent in the order of the nodes array
        self.global_num = global_num

    
# takes an empty nodes list, degree of freedom per node, and x and y list of locations 
# and returns a list of nodes of type Node, list is ordered
def nodes_List(nodes, n_ed, xnodes, ynodes, znodes=None):
    i = 0
    if znodes == None:
        for y in ynodes:
            for x in xnodes:
                # Node numbering starts at 1 indexing, hence the i+1
                nodes[i] = Node(n_ed, np.array([x, y]), i+1)
                i = i + 1
    return nodes





# returns an nparray of Bilinear Quadrilateral Elements for Beam Conventional Ordering
# nodes is an nparray of type Node already ordered globally
# in 2D, there are only 4 nodes per element
def BQE(nodes, xnum, ynum):
    # Calculate number of elements: (xnum-1)*(ynum-1)
    num_elements = (xnum-1)*(ynum-1)
    
    # Initialize array to store Element objects
    elements = np.zeros(num_elements, dtype=object)

    # Loop through the grid to form quadrilateral elements
    element_idx = 1
    for j in range(ynum-1): # rows
        for i in range(xnum-1): # columns
            # Node indices for the quadrilateral (counter-clockwise order)
            # Assuming nodes are ordered row-wise: [0, 1, ..., xnum-1] for first row, etc
            node1 = j * xnum + i        # Bottom Left
            node2 = j * xnum + (i + 1)  # Bottom right
            node3 = (j + 1) * xnum + (i + 1) # Top right
            node4 = (j+ 1) * xnum + i       # Top left
    
            # Create array of 4 nodes for this element
            element_nodes = np.array([nodes[node1], nodes[node2], nodes[node3], nodes[node4]])

            # Create Element object
            elements[element_idx - 1] = Element(element_nodes, element_idx)
            element_idx += 1
    return elements


# nodes is nparray of objet Nodes
# elements in nparray of object Element
# n_ed is dof per node
def generate_ID_IEN(nodes, elements, n_ed):
    # Total number of nodes
    n_n = len(nodes)
    # Total number of elements
    n_e = len(elements)
    # Number of nodes per element (4 for quadrilateral)
    n_en = 4

    # Initialize ID array: shape (n_ed, n_n)
    ID = np.zeros((n_ed, n_n), dtype=int)
    
    # Assign DOF numbers
    dof_count = 1  # Start DOF numbering at 1
    for j in range(n_n):
        for i in range(n_ed):
            ID[i, j] = dof_count
            dof_count += 1

    # Initialize IEN array: shape (n_en, n_e)
    IEN = np.zeros((n_en, n_e), dtype=int)
    
    # Populate IEN array
    for k in range(n_e):
        # Get global node numbers for element k (1-based, so subtract 1 for 0-based array)
        IEN[:, k] = [node.global_num for node in elements[k].nodes]

    return ID, IEN



# Applies boundary conditions for a cantilever beam (fixed at x=0)
def apply_boundary_conditions(nodes, ID):
    """
    Apply boundary conditions: fixed at x=0 (all DOFs set to 0).
    Renumber remaining DOFs contiguously to account for reduced equations.
    """
    n_ed, n_n = ID.shape
    # Initialize new ID array
    new_ID = np.zeros((n_ed, n_n), dtype=int)
    
    # Mark fixed DOFs (x=0) as 0
    fixed_dofs = set()
    for j in range(n_n):
        if abs(nodes[j].point[0]) < 1e-6:  # Nodes at x=0
            new_ID[0, j] = 0  # x-DOF
            new_ID[1, j] = 0  # y-DOF
            fixed_dofs.add((0, j))
            fixed_dofs.add((1, j))
    
    # Renumber free DOFs contiguously
    dof_count = 1
    for j in range(n_n):
        for i in range(n_ed):
            if (i, j) not in fixed_dofs:
                new_ID[i, j] = dof_count
                dof_count += 1
    
    return new_ID


    