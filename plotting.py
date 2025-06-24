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