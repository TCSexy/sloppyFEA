import numpy as np
from matplotlib import pyplot as plt
import mesh as msh


# plots the nodes of the beam. Additionally takes in length and height of beam
# to scale the plot nicely
def nodes_Plot(nodes, l, h):
    # Extract x and y coordinates from nodes for scatter plot
    x_coords = [node.point[0] for node in nodes]
    y_coords = [node.point[1] for node in nodes]

    plt.scatter(x_coords, y_coords, marker='o', color='blue', label='Nodes')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Node Mesh for Beam')
    plt.xlim(0, 1.5*l)
    plt.ylim(-0.5*h, 1.5*h)
    plt.show()


# Plot element boundaries
def element_Plot(elements):
    for element in elements:
        # Get x and y coordinates of the four nodes in counterclockwise order
        x = [node.point[0] for node in element.nodes]
        y = [node.point[1] for node in element.nodes]
        # Close the loop by repeating the first node
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, 'r-', linewidth=1, label='Elements' if element.global_num == 1 else "")
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Node and Element Mesh for Beam')
    plt.legend()
    plt.show()


# Plots both nodes and elements in a single figure.
def mesh_Plot(nodes, elements, l, h):
    # Plot nodes
    if len(elements) < 100:
        x_coords = [node.point[0] for node in nodes]
        y_coords = [node.point[1] for node in nodes]
        plt.scatter(x_coords, y_coords, marker='o', color='blue', label='Nodes')

    # Plot element boundaries
    for element in elements:
        x = [node.point[0] for node in element.nodes]
        y = [node.point[1] for node in element.nodes]
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, 'r-', linewidth=1, label='Elements' if element.global_num == 1 else "")

        # Optional: Label elements with their global number at centroid
        if len(elements) < 100:
            centroid_x = sum(x[:-1]) / 4  # Average of x coordinates (excluding repeated point)
            centroid_y = sum(y[:-1]) / 4  # Average of y coordinates
            plt.text(centroid_x, centroid_y, str(element.global_num), fontsize=8, ha='center', va='center', color='black')

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Node and Element Mesh for Beam')
    plt.xlim(0, 1.5 * l)
    plt.ylim(-0.5 * h, 1.5 * h)
    plt.legend()
    plt.show()



def deformed_plot1(nodes, l, h, scale=1e3):
    """
    Plot the deformed and original mesh.
    
    Parameters:
        nodes (np.ndarray): Array of Node objects with displacements.
        l (float): Beam length in meters.
        h (float): Beam height in meters.
        scale (float): Displacement scaling factor for visualization.
    """
    plt.figure()
    x_orig = [node.point[0] for node in nodes]
    y_orig = [node.point[1] for node in nodes]
    x_def = [node.point[0] + scale * node.disp[0] for node in nodes]
    y_def = [node.point[1] + scale * node.disp[1] for node in nodes]
    
    plt.scatter(x_orig, y_orig, marker='o', color='blue', label='Original Nodes')
    plt.scatter(x_def, y_def, marker='o', color='red', label='Deformed Nodes')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Deformed vs Original Mesh')
    plt.xlim(0, 1.5 * l)
    plt.ylim(-0.5 * h, 1.5 * h)
    plt.legend()
    plt.show()


def deformed_plot2(nodes, l, h, xnum, ynum):
    """
    Plot the outline of the original and deformed beam with automatic scaling.
    
    Parameters:
        nodes (np.ndarray): Array of Node objects with displacements.
        l (float): Beam length in meters.
        h (float): Beam height in meters.
        xnum (int): Number of nodes in x-direction.
        ynum (int): Number of nodes in y-direction.
    """
    plt.figure()
    
    # Compute maximum displacement for auto-scaling
    max_displacement = max(max(abs(node.disp[0]), abs(node.disp[1])) for node in nodes)
    
    # Auto-scale: Amplify small displacements, proportional for larger ones
    c = 1  # Scaling constant to amplify displacements
    min_displacement = 1e-6  # Minimum displacement to avoid excessive scaling
    scale = c * l / max(max_displacement, min_displacement)
    # Cap scale to prevent excessive deformation (max plotted displacement <= 0.5 * l)
    if scale * max_displacement > 0.5 * l:
        scale = 0.5 * l / max_displacement
    
    # Round scale to 2 significant figures for display
    scale_display = f"{scale:.2e}"
    
    # Identify boundary nodes (counterclockwise: bottom, right, top, left)
    boundary_nodes = []
    # Bottom edge (y=0, x from 0 to l)
    for i in range(xnum):
        idx = i  # First row: nodes 0 to xnum-1
        boundary_nodes.append(nodes[idx])
    # Right edge (x=l, y from 0 to h)
    for j in range(1, ynum):  # Start from 1 to avoid duplicating corner
        idx = xnum - 1 + j * xnum  # Last column
        boundary_nodes.append(nodes[idx])
    # Top edge (y=h, x from l to 0)
    for i in range(xnum-2, -1, -1):  # Reverse order, skip last node
        idx = (ynum-1) * xnum + i  # Last row
        boundary_nodes.append(nodes[idx])
    # Left edge (x=0, y from h to 0)
    for j in range(ynum-2, 0, -1):  # Reverse order, skip bottom-left
        idx = j * xnum  # First column
        boundary_nodes.append(nodes[idx])
    # Close the loop by adding the first node again
    boundary_nodes.append(boundary_nodes[0])
    
    # Original outline
    x_orig = [node.point[0] for node in boundary_nodes]
    y_orig = [node.point[1] for node in boundary_nodes]
    plt.plot(x_orig, y_orig, 'b-', linewidth=2, label='Original Outline')
    
    # Deformed outline
    x_def = [node.point[0] + scale * node.disp[0] for node in boundary_nodes]
    y_def = [node.point[1] + scale * node.disp[1] for node in boundary_nodes]
    plt.plot(x_def, y_def, 'r-', linewidth=2, label='Deformed Outline')
    
    # Annotate plot with scale factor
    plt.text(0.05 * l, 1.3 * h, f'Displacement Scale: {scale_display}', fontsize=10, color='black')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Original vs Deformed Beam Outline')
    plt.xlim(0, 1.5 * l)
    plt.ylim(-0.5 * h, 1.5 * h)
    plt.legend()
    plt.grid(True)
    plt.show()


def deformed_plot(nodes, l, h, xnum, ynum):
    """
    Plot the outline of the original and deformed beam with adaptive auto-scaling.
    
    Parameters:
        nodes (np.ndarray): Array of Node objects with displacements.
        l (float): Beam length in meters.
        h (float): Beam height in meters.
        xnum (int): Number of nodes in x-direction.
        ynum (int): Number of nodes in y-direction.
    """
    plt.figure()
    
    # Compute maximum displacement for auto-scaling
    max_displacement = max(max(abs(node.disp[0]), abs(node.disp[1])) for node in nodes)
    
    # Auto-scale: Adaptive scaling to emphasize larger displacements
    k = 0.1  # Base scaling constant
    ref_displacement = 1e-4  # Reference displacement for small loads
    alpha = 0.7  # Exponent to control scaling sensitivity
    scale = k * l / max(max_displacement, ref_displacement)**alpha
    # Cap scale to prevent excessive deformation (max plotted displacement <= l)
    c_max = 1.0  # Maximum plotted displacement as fraction of beam length
    if scale * max_displacement > c_max * l:
        scale = c_max * l / max_displacement
    
    # Round scale to 2 significant figures for display
    scale_display = f"{scale:.2e}"
    
    # Identify boundary nodes (counterclockwise: bottom, right, top, left)
    boundary_nodes = []
    # Bottom edge (y=0, x from 0 to l)
    for i in range(xnum):
        idx = i  # First row: nodes 0 to xnum-1
        boundary_nodes.append(nodes[idx])
    # Right edge (x=l, y from 0 to h)
    for j in range(1, ynum):  # Start from 1 to avoid duplicating corner
        idx = xnum - 1 + j * xnum  # Last column
        boundary_nodes.append(nodes[idx])
    # Top edge (y=h, x from l to 0)
    for i in range(xnum-2, -1, -1):  # Reverse order, skip last node
        idx = (ynum-1) * xnum + i  # Last row
        boundary_nodes.append(nodes[idx])
    # Left edge (x=0, y from h to 0)
    for j in range(ynum-2, 0, -1):  # Reverse order, skip bottom-left
        idx = j * xnum  # First column
        boundary_nodes.append(nodes[idx])
    # Close the loop by adding the first node again
    boundary_nodes.append(boundary_nodes[0])
    
    # Original outline
    x_orig = [node.point[0] for node in boundary_nodes]
    y_orig = [node.point[1] for node in boundary_nodes]
    plt.plot(x_orig, y_orig, 'b-', linewidth=2, label='Original Beam')
    
    # Deformed outline
    x_def = [node.point[0] + scale * node.disp[0] for node in boundary_nodes]
    y_def = [node.point[1] + scale * node.disp[1] for node in boundary_nodes]
    plt.plot(x_def, y_def, 'r-', linewidth=2, label='Deformed Beam')
    
    # Annotate plot with scale factor
    plt.text(0.05 * l, 1.3 * h, f'Displacement Scale: {scale_display}', fontsize=10, color='black')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Original vs Deformed Beam Outline')
    plt.xlim(0, 1.5 * l)
    plt.ylim(-0.5 * h, 1.5 * h)
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_theory(nodes, xnum, l, t, theory):
    """
    Compare FEA displacements to theoretical predictions for a single thickness.
    
    Parameters:
        nodes (np.ndarray): Array of Node objects with displacements.
        xnum (int): Number of nodes in x-direction.
        ynum (int): Number of nodes in y-direction.
        l (float): Beam length in meters.
        h (float): Beam height in meters.
        t (float): Beam thickness in meters.
        E (float): Young's modulus (Pa).
        I (float): Moment of inertia for the given thickness.
        theory (callable): Lambda function for theoretical displacement, expecting x and I.
    """
    # Extract bottom edge nodes (y=0, indices 0 to xnum-1)
    x_coords = np.linspace(0, l, xnum)  # x-coordinates of bottom edge nodes
    fea_displacements = np.array([nodes[i].disp[1] for i in range(xnum)])  # y-displacements

    # Compute theoretical displacements for the given thickness
    theo_displacements = theory(x_coords)

    # Plot comparison
    plt.figure()
    plt.plot(x_coords, fea_displacements, 'k-o', label='FEA', markersize=5, linewidth=2)
    plt.plot(x_coords, theo_displacements, 'b--', label=f'Theory (t={t} m)')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y-Displacement (meters)')
    plt.title('FEA vs Theoretical Displacements (Bottom Edge)')
    plt.legend()
    plt.show()

    # Compute errors
    abs_error = np.abs(fea_displacements - theo_displacements)
    max_abs_error = np.max(abs_error)
    rmse = np.sqrt(np.mean(abs_error**2))
    
    # Compute percent error, avoiding division by zero
    percent_error = np.abs((fea_displacements[-1] - theo_displacements[-1]) / (np.abs(theo_displacements[-1]))) * 100
    
    print(f'Thickness t={t} m:')
    print(f'  Max Absolute Error: {max_abs_error:.2e} m')
    print(f'  RMSE: {rmse:.2e} m')
    print(f'  Percent Error: {percent_error:.2f}%')