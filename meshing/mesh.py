import numpy as np

# Beam mesh takes in the length, width, and number of elements, and returns
# two nparrays on x and y locations which correspond to node points
def beamMesh(l, w, xnodes, ynodes):
    xpoints = np.linspace(0, l, xnodes)
    ypoints = np.linspace(0, w, ynodes)
    print(xnodes)
    
