from typing import Dict, List, Optional, Tuple, Callable
import os
import vedo
import trimesh
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pytorch3d.transforms import *

from models.submodules import Tooth_Centering


class TeethDataset(Dataset):
    def __init__(self,
                df: pd.DataFrame,
                sample_num: int = 512,
                transform: Optional[Callable] = None
                ):
        super(TeethDataset, self).__init__()
        self.df = df
        self.sample_num = sample_num
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        data = dict()
        thing_from_df = self.df.values[idx]
        mesh = vedo.Mesh(thing_from_df[0])

        # data["X"], data["X_v"], data["C"] = Tooth_Centering().get_pcds_and_centers(mesh, self.sample_num)

        # save data as numpy for debugging purposes
        # np.save("F:\DataSlow\TeethAlignment\debug\X.npy", data["X"].numpy())
        # np.save("F:\DataSlow\TeethAlignment\debug\X_v.npy", data["X_v"].numpy())
        # np.save("F:\DataSlow\TeethAlignment\debug\C.npy", data["C"].numpy())

        # load the saved numpy data for debugging purposes
        data["X"] = torch.from_numpy(np.load("F:\DataSlow\TeethAlignment\debug\X.npy"))
        data["X_v"] = torch.from_numpy(np.load("F:\DataSlow\TeethAlignment\debug\X_v.npy"))
        data["C"] = torch.from_numpy(np.load("F:\DataSlow\TeethAlignment\debug\C.npy"))


        data["target_X"], data["target_X_v"] = data["X"].clone(), data["X_v"].clone()
        data["6dof"] = torch.zeros(size=(data["X_v"].shape[0], 6))

        if self.transform is not None:
            data = self.transform(data)


        data["C"] = torch.cat([torch.from_numpy(trimesh.PointCloud(data["X_v"][i, :, :].numpy()).centroid).unsqueeze(0) for i in range(data["C"].shape[0])], dim=0)
        data["C"] = torch.cat((data["C"],
                               torch.from_numpy(trimesh.PointCloud(data["X"][:(int)(data["X"].shape[0]/2), :].numpy()).centroid).unsqueeze(0),
                               torch.from_numpy(trimesh.PointCloud(data["X"][(int)(data["X"].shape[0]/2):, :].numpy()).centroid).unsqueeze(0)),
                              dim=0).float()

        return data
