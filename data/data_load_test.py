import vedo

import trimesh
import csv


import trimesh
from vedo import Points, show


# Load the STL model file
stl_file = 'F:/DataSlow/TeethAlignment/debug/dataset_csvextract_teeth_without_gingiva.stl'
mesh = trimesh.load(stl_file)

# convert to point cloud
samples = trimesh.sample.sample_surface_even(mesh, 5000)[0]
point_cloud = trimesh.PointCloud(samples)

points_vedo = Points(point_cloud.vertices, r=5)

# set red color for all points
#points_vedo.c("red")
#points_vedo.cmap("jet", point_cloud.vertices)


# color the point cloud based on the z coordinate
points_vedo.cmap("bone", point_cloud.vertices[:,1])

# mesh = trimesh.creation.icosphere()
# mesh.visual.face_colors = [200, 200, 250, 100]
# n = mesh.vertices.shape[0]
#
# #Assign a color based on a scalar and a color map
# pc1 = Points(mesh.vertices, r=10)
# pc1.cmap("jet", list(range(n)))
#
# pc2 = Points(mesh.vertices, r=10)
# pc2.cmap("viridis", mesh.vertices[:,2])

# Draw result on N=2 sync'd renderers
## Draw result on N=2 sync'd renderers
#show([(mesh,pc1), (mesh,pc2)], N=2, axes=1)
show([(points_vedo)], N=1, axes=1)



#make all the points red
#points_vedo.c("red")

show(points_vedo, axes=1)

# show scene with light, axes, and point cloud
scene = trimesh.Scene()
point_cloud.visual.vertex_colors = trimesh.visual.random_color()
scene.add_geometry(point_cloud)
scene.add_geometry(trimesh.creation.axis(axis_length=10, axis_radius=0.001))
#scene.lights = trimesh.scene.lighting.autolight(scene)[0]
scene.show()


csv_file = 'F:/DataSlow/TeethAlignment/dataset_csvKristen-upperjaw.csv'

#get the first row of the csv file
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    row = next(reader)
    row = next(reader)

    # get first item in row
    x = row[0]
    print(x)
    mesh = vedo.Mesh(x)





# Save the vertices of the mesh to a CSV file
csv_file = 'F:/DataSlow/TeethAlignment/dataset_csvKristen-upperjaw.csv'
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y', 'z'])  # Write header

    # Write the vertices to the CSV file
    for vertex in mesh.vertices:
        writer.writerow(vertex)


# Convert the mesh to a vedo mesh
mesh = vedo.Mesh(mesh)



#save the mesh as a csv file
#mesh.write('F:/DataSlow/TeethAlignment/dataset_csvKristen-upperjaw.csv')

# Load the csv file with csv reader
import pandas as pd
csv = pd.read_csv('F:/DataSlow/TeethAlignment/dataset_csvKristen-upperjaw.csv')

print(csv)