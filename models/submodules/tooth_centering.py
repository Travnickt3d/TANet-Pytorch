from typing import Dict, List, Optional, Tuple, Callable
import vedo
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.utils.dlpack

class Tooth_Centering(nn.Module):
    def __init__(self):
        super(Tooth_Centering, self).__init__()

    # CPU Version
    def get_pcds_and_centers(self, mesh: vedo.Mesh, sample_num=512):
        "trimesh/util.py -> len(index) => index.size"

        # all teeth labels
        labels = np.unique(mesh.celldata["Label"])

        # for label in np.unique(mesh.celldata["Label"]):
        #     print("Label: ", label, " count: ", np.sum(mesh.celldata["Label"] == label))

        #labels_count = len(labels)
        labels_count = 28
        #print("getitem labels: ", labels)

        #max_label = mesh.celldata["Label"].max()
        #total_points_num = sample_num*mesh.celldata["Label"].max()
        #C = np.zeros((max_label, 3), dtype=np.float32)

        total_points_num = sample_num*labels_count
        C = np.zeros((labels_count, 3), dtype=np.float32)

        X = np.zeros((total_points_num, 3), dtype=np.float32)
        trimesh_tmp = mesh.to_trimesh()
        #trimesh_tmp.export(f"F:\DataSlow\TeethAlignment\debug\mesh_before_teeth_sampling.stl")

        #for i in range(max_label):
        for label, i in zip(labels, range(labels_count)):

            #indices = np.where(mesh.celldata["Label"]==i+1)
            indices = np.where(mesh.celldata["Label"]==label)

            if len(indices[0]) == 0:
                # fill with 0 points
                X[i*sample_num:(i*sample_num+sample_num)] = np.zeros((sample_num, 3), dtype=np.float32)
            else:
                #indices = indices[0]
                tmp = trimesh_tmp.submesh(indices, append=True)
                #tmp.export(f"F:\DataSlow\TeethAlignment\debug\submesh_{i}.stl")
                #tmp = mesh.to_trimesh().submesh(np.where(mesh.celldata["Label"]==i+1)[0], append=True)
                tooth = trimesh.sample.sample_surface_even(tmp, sample_num)[0]
                C[i] = trimesh.points.PointCloud(vertices=tooth).centroid
                X[i*sample_num:(i*sample_num+sample_num)] = tooth


        #X_v = X.reshape(max_label, sample_num, 3)
        X_v = X.reshape(labels_count, sample_num, 3)

        return torch.from_numpy(X), torch.from_numpy(X_v), torch.from_numpy(C)

    @torch.no_grad()
    def forward(self, X: Dict[str, torch.Tensor], device: torch.device)->Dict[str, torch.Tensor]:
        C = dict()
        C["X_v"] = torch.zeros(X["X_v"].shape, dtype=torch.float32, device=device)
        for batch_idx in range(X["X_v"].shape[0]):
            for tooth_idx in range(X["X_v"].shape[1]):
                C["X_v"][batch_idx][tooth_idx] = X["X_v"][batch_idx][tooth_idx]-X["C"][batch_idx][tooth_idx]
        C["X"] = C["X_v"].clone().reshape(X["X"].shape)
        C["C"] = X["C"]
        C["6dof"] = X["6dof"]
        return C
