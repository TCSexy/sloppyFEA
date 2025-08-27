import numpy as np
import mesh as msh


# Takes in an element and returns the element stiffness matrix
def element_stiffness(element, t, E=70e9, nu=0.33):
    n_en = len(element.nodes)
    n_ed = element.nodes[0].n_ed  # DOFs per node (2 for 2D)
    k = np.zeros((n_en * n_ed, n_en * n_ed))  # 8x8 for 4 nodes, 2 DOFs each
    
    # Gaussian quadrature points and weights
    n_int = 4
    ksi = np.array([-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)])
    eta = np.array([-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    weights = np.ones(4)  # Weight = 1 for each point
    
    # Shape function derivatives (dN/dksi, dN/deta)
    dN = np.array([
        # [dN_i/dksi, dN_i/deta] for each node at each quadrature point
        [[(-1/4)*(1-eta[l]), (-1/4)*(1-ksi[l])] for l in range(n_int)],  # N1
        [[(1/4)*(1-eta[l]), (-1/4)*(1+ksi[l])] for l in range(n_int)],   # N2
        [[(1/4)*(1+eta[l]), (1/4)*(1+ksi[l])] for l in range(n_int)],    # N3
        [[(-1/4)*(1+eta[l]), (1/4)*(1-ksi[l])] for l in range(n_int)]     # N4
    ])  # Shape: (4 nodes, 4 points, 2 derivatives)
    
    # Constitutive matrix (plane stress)
    C = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    
    # Loop over quadrature points
    for l in range(n_int):
        # Shape function derivatives at quadrature point l
        dN_ksi = np.array([dN[i][l][0] for i in range(4)])
        dN_eta = np.array([dN[i][l][1] for i in range(4)])
        
        # Nodal coordinates
        x = np.array([node.point[0] for node in element.nodes])
        y = np.array([node.point[1] for node in element.nodes])
        
        # Jacobian matrix (transposed convention)
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_ksi, x)  # dx/dksi
        J[0, 1] = np.dot(dN_eta, x)  # dx/deta
        J[1, 0] = np.dot(dN_ksi, y)  # dy/dksi
        J[1, 1] = np.dot(dN_eta, y)  # dy/deta
        detJ = np.linalg.det(J)
        
        # Inverse Jacobian
        invJ = np.linalg.inv(J)
        
        # Derivatives of shape functions w.r.t. x, y
        dN_xy = np.dot(invJ.T, np.array([dN_ksi, dN_eta]))  # [dN/dx; dN/dy]
        
        # Strain-displacement matrix B
        B = np.zeros((3, n_en * n_ed))
        for i in range(n_en):
            B[0, 2*i] = dN_xy[0, i]    # dN_i/dx
            B[1, 2*i + 1] = dN_xy[1, i]  # dN_i/dy
            B[2, 2*i] = dN_xy[1, i]      # dN_i/dy (shear)
            B[2, 2*i + 1] = dN_xy[0, i]  # dN_i/dx (shear)
        
        # Element stiffness: k += B^T * C * B * detJ * weight
        k += t * weights[l] * detJ * np.dot(B.T, np.dot(C, B))
    
    return k


# Takes in the global stiffness matrix and an element stiffness matrix, along
# with the necessary ID and IEN arrays and updates global stiffness matrix
# IEN_col is not full IEN array, just the column associated with the element
def K_assemble(K, k, ID, IEN_col):
    """
    Assemble element stiffness matrix k into global stiffness matrix K.
    K: Global stiffness matrix (n_eq x n_eq)
    k: Element stiffness matrix (n_en * n_ed x n_en * n_ed)
    ID: DOF mapping array (n_ed x n_n)
    IEN_col: Global node numbers for the element (n_en)
    """
    n_en = len(IEN_col)  # Number of nodes per element (4)
    n_ed = 2  # DOFs per node (hardcoded for 2D)
    
    for i in range(n_en):
        for j in range(n_en):
            for m in range(n_ed):
                for n in range(n_ed):
                    # Get global DOF numbers from ID array
                    global_dof_i = ID[m, IEN_col[i] - 1]  # -1 for 0-based indexing
                    global_dof_j = ID[n, IEN_col[j] - 1]
                    # Only add to K if both DOFs are non-zero (not fixed)
                    if global_dof_i != 0 and global_dof_j != 0:
                        K[global_dof_i - 1, global_dof_j - 1] += k[i * n_ed + m, j * n_ed + n]
    
    return K
    

# ID array takes degree of freedom and global node number and returns global equation number
# IEN array takes global element number and local node number and returns global node number
# elements is nparray of all elements
# n_eq is number of global equations
# Set up the global stiffness matrix and force vector
def solution_setup(ID, IEN, elements, n_eq, h, t, traction, E=70e9, nu=0.33):
    """
    ID: DOF mapping array (n_ed x n_n)
    IEN: Element-node connectivity array (n_en x n_e)
    elements: Array of Element objects
    n_eq: Number of global equations
    h: Beam height
    traction: Traction vector [t_x, t_y] (N/m)
    E, nu, t: Material properties and thickness
    Returns: K (global stiffness matrix), F (force vector)
    """
    K = np.zeros((n_eq, n_eq), dtype=float)
    F = np.zeros(n_eq, dtype=float)
    
    # Assemble stiffness matrix
    for i in range(len(elements)):
        k = element_stiffness(elements[i], t, E, nu)
        K = K_assemble(K, k, ID, IEN[:, elements[i].global_num - 1])
    

    '''
    Can eventually modularize the following force vector calculation in its own
    function. The nice thing about this specific case is that the fixed points are
    0 displacement, so all dirichlet terms are 0 and we can ignore them
    '''

    # Compute force vector for traction on top edge (y = h, nodes 3 and 4)
    n_ed = 2  # DOFs per node
    for element in elements:
        if abs(element.nodes[2].point[1] - h) < 1e-6 and abs(element.nodes[3].point[1] - h) < 1e-6:
            # Top edge: nodes 3 and 4 (indices 2 and 3 in element.nodes)
            x1, y1 = element.nodes[2].point  # Node 3
            x2, y2 = element.nodes[3].point  # Node 4
            L_e = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Edge length
            
            # 1D Gaussian quadrature (2 points for linear edge)
            xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            weights = np.array([1.0, 1.0])
            f_e = np.zeros(2 * n_ed)  # Force vector for 2 nodes * 2 DOFs
            
            for l in range(len(xi)):
                # 1D linear shape functions for edge (nodes 3 and 4)
                N = np.array([(1 + xi[l]) / 2, (1 - xi[l]) / 2])  # [N3, N4] following convention from k matrix
                # Shape function matrix for 2D DOFs
                N_mat = np.zeros((2, 2 * n_ed))
                N_mat[0, 0] = N[0]  # Node 3, x-DOF
                N_mat[1, 1] = N[0]  # Node 3, y-DOF
                N_mat[0, 2] = N[1]  # Node 4, x-DOF
                N_mat[1, 3] = N[1]  # Node 4, y-DOF
                
                # Jacobian: ds/dxi = L_e / 2
                J = L_e / 2
                # Integrate: f_e += N^T * t * J * w
                f_e += weights[l] * J * np.dot(N_mat.T, traction)
            
            # Assemble into global force vector
            for j in range(2):  # Nodes 3 and 4
                global_node = element.nodes[j + 2].global_num
                for dof in range(n_ed):
                    global_dof = ID[dof, global_node - 1]
                    if global_dof != 0:
                        F[global_dof - 1] += f_e[j * n_ed + dof]

    return K, F