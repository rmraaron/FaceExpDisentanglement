"""Divide BU3DFE into training set and test set for DataLoader"""


import logging
import torch
from torch.utils.data import Dataset
import numpy as np
from path import *
from psbody.mesh import MeshViewers, Mesh
import time
import platform


EXPRESSION_MAPPING = {'NE': 0, 'AN': 1, 'DI': 2, 'FE': 3, 'HA': 4, 'SA': 5, 'SU': 6}
GENDER_MAPPING = {'F': 0, 'M': 1}
RACE_MAPPING = {'AE': 0, "AM": 1, "BL": 2, "IN": 3, "LA": 4, "WH": 5}


class BU3DFE(Dataset):
    def __init__(self, partition='train', include_neutral=True, always_sample_same_id=False, sort=False):
        # 156 is the longest vertex distance from the origin, to normalise.
        vertices = np.load(DATASET_PATH + 'BU3DFE_ver_' + partition + '_reg.npy') / BU3DFE_NORMALISE
        labels = np.load(DATASET_PATH + 'BU3DFE_label_' + partition + '.npy', allow_pickle=True)[()]

        expressions = np.vectorize(EXPRESSION_MAPPING.get)(labels['expression'])
        expression_levels = np.array(labels['expression_level'])
        gender = np.vectorize(GENDER_MAPPING.get)(labels['subject_gender'])
        race = np.vectorize(RACE_MAPPING.get)(labels['race'])
        subject_id = np.array(labels['subject_id'])

        if sort:
            sort_indices = gender * 1000 + subject_id * 10 + expressions + expression_levels / 4
            sort_indices = np.argsort(sort_indices)
            vertices = vertices[sort_indices]
            expressions = expressions[sort_indices]
            expression_levels = expression_levels[sort_indices]
            gender = gender[sort_indices]
            race = race[sort_indices]
            subject_id = subject_id[sort_indices]

        if partition == "train":
            # Save normalised mean face.
            mean_face = np.average(np.array(vertices), axis=0)
            np.save(DATASET_PATH + 'BU3DFE_mean_face', mean_face)
            if os.path.exists(DATASET_PATH + "BU3DFE_exp_gt_dict.pt"):
                logging.info("Using existing exp gt dict.")
                self.expression_gt_dict = torch.load(DATASET_PATH + "BU3DFE_exp_gt_dict.pt")
            else:
                self.expression_gt_dict = {}
                mesh_viewer = MeshViewers(shape=(1, 2))
                # i: 7 expressions including neutral
                for i in range(7):
                    if i not in self.expression_gt_dict:
                        self.expression_gt_dict[i] = {}
                    # m: male and female
                    for m in range(2):
                        self.expression_gt_dict[i][m] = {}
                        if i == 0:
                            exp_average_gt = np.average(vertices[np.logical_and(expressions == i, gender == m)], axis=0)
                            self.expression_gt_dict[i][m][0] = torch.from_numpy(exp_average_gt).to(torch.float32)
                            bu3dfe_faces = np.load(DATASET_PATH + "BU3DFE_face.npy")
                            mesh_exp_true = Mesh(f=bu3dfe_faces)
                            mesh_exp_true.v = exp_average_gt
                            mesh_mean_face = Mesh(f=bu3dfe_faces)
                            mesh_mean_face.v = mean_face
                            mesh_viewer[0][0].set_dynamic_meshes([mesh_exp_true])
                            mesh_viewer[0][1].set_dynamic_meshes([mesh_mean_face])
                            time.sleep(1)
                        else:
                            # j: 4 expression levels
                            for j in range(1, 5):
                                exp_average_gt = np.average(vertices[np.logical_and(expressions == i, np.logical_and(expression_levels == j, gender == m))], axis=0)
                                self.expression_gt_dict[i][m][j] = torch.from_numpy(exp_average_gt).to(torch.float32)
                                bu3dfe_faces = np.load(DATASET_PATH + "BU3DFE_face.npy")
                                mesh_exp_true = Mesh(f=bu3dfe_faces)
                                mesh_exp_true.v = exp_average_gt
                                mesh_mean_face = Mesh(f=bu3dfe_faces)
                                mesh_mean_face.v = mean_face
                                mesh_viewer[0][0].set_dynamic_meshes([mesh_exp_true])
                                mesh_viewer[0][1].set_dynamic_meshes([mesh_mean_face])
                                time.sleep(1)
                torch.save(self.expression_gt_dict, DATASET_PATH + "BU3DFE_exp_gt_dict.pt")
        else:
            self.expression_gt_dict = torch.load(DATASET_PATH + "BU3DFE_exp_gt_dict.pt")

        def move_to(obj, device):
            if torch.is_tensor(obj):
                return obj.to(device)
            elif isinstance(obj, dict):
                res = {}
                for k, v in obj.items():
                    res[k] = move_to(v, device)
                return res
            elif isinstance(obj, list):
                res = []
                for v in obj:
                    res.append(move_to(v, device))
            else:
                raise TypeError("Invalid type for move_to")

        if platform.system() == "Linux":
            self.vertices = torch.from_numpy(vertices).to(torch.float32).to(device)
            self.expressions = torch.from_numpy(expressions).to(device)
            self.expression_level = torch.from_numpy(expression_levels).to(device)
            self.subject_id = torch.from_numpy(subject_id).to(device)
            self.gender = torch.from_numpy(gender).to(torch.bool).to(device)
            self.race = torch.from_numpy(race).to(device)
            self.always_sample_same_id = always_sample_same_id
            self.exp_gt_dict = move_to(self.expression_gt_dict, device)
            self.expression_gt_dict = self.exp_gt_dict
        else:
            self.vertices = torch.from_numpy(vertices).to(torch.float32)
            self.expressions = torch.from_numpy(expressions)
            self.expression_level = torch.from_numpy(expression_levels)
            self.subject_id = torch.from_numpy(subject_id)
            self.gender = torch.from_numpy(gender).to(torch.bool)
            self.race = torch.from_numpy(race)
            self.always_sample_same_id = always_sample_same_id

        if not include_neutral:
            self.vertices = self.vertices[expressions != 0]
            # EXPRESSION_MAPPING: {'AN': 0, 'DI': 1, 'FE': 2, 'HA': 3, 'SA': 4, 'SU': 5}
            self.expressions = self.expressions[expressions != 0] - 1
            self.expression_level = self.expression_level[expressions != 0]
            self.subject_id = self.subject_id[expressions != 0]
            self.gender = self.gender[expressions != 0]
            self.race = self.race[expressions != 0]

    @property
    def subject_ids(self):
        return self.gender * 100 + self.subject_id

    def __len__(self):
        return self.vertices.shape[0]

    def __getitem__(self, index):
        if not self.always_sample_same_id:
            same_id_label = torch.randint(2, (1,)) == 1
        else:
            same_id_label = torch.tensor([True], dtype=torch.bool)
        sample_id_idx = self.sample_id(self.gender[index], self.subject_id[index], same_id_label.item())
        same_exp_label = torch.randint(2, (1,)) == 1
        sample_exp_idx = self.sample_exp(self.expressions[index], same_exp_label.item())
        neutral_id_idx = self.get_neutral(self.gender[index], self.subject_id[index])
        exp_lvl = self.expression_level[index].item()
        exp_v = self.expression_gt_dict[self.expressions[index].item()][self.gender[index].item()][exp_lvl]

        if platform.system() == "Linux":
            return self.vertices[index], \
                   self.expressions[index], \
                   self.expression_level[index], \
                   self.gender[index], \
                   self.race[index], \
                   self.subject_id[index], \
                   self.vertices[sample_id_idx], \
                   same_id_label.to(device, dtype=torch.long).squeeze(-1), \
                   self.vertices[sample_exp_idx], \
                   same_exp_label.to(device, dtype=torch.long).squeeze(-1), \
                   self.vertices[neutral_id_idx], \
                   exp_v
        else:
            return self.vertices[index].to(device), \
                self.expressions[index].to(device),\
                self.expression_level[index].to(device),\
                self.gender[index].to(device),\
                self.race[index].to(device), \
                self.subject_id[index].to(device), \
                self.vertices[sample_id_idx].to(device), \
                same_id_label.to(device, dtype=torch.long).squeeze(-1), \
                self.vertices[sample_exp_idx].to(device), \
                same_exp_label.to(device, dtype=torch.long).squeeze(-1), \
                self.vertices[neutral_id_idx].to(device), \
                exp_v.to(device)

    def sample_id(self, gender, id_, same):
        if same:
            id_idx = torch.nonzero(torch.logical_and(self.gender == gender, self.subject_id == id_))
        else:
            id_idx = torch.nonzero(torch.logical_or(self.gender != gender, self.subject_id != id_))
        rand_idx = torch.randint(id_idx.shape[0], (1,))

        return id_idx[rand_idx].item()

    def sample_exp(self, exp, same):
        if same:
            exp_idx = torch.nonzero(self.expressions == exp)
        else:
            exp_idx = torch.nonzero(self.expressions != exp)
        rand_exp_idx = torch.randint(exp_idx.shape[0], (1,))

        return exp_idx[rand_exp_idx].item()

    def get_neutral(self, gender, id_):
        id_idx = torch.nonzero(torch.logical_and(self.gender == gender, torch.logical_and(self.subject_id == id_, self.expressions == 0)))
        return id_idx.item()

    @property
    def vertices_num(self):
        return self.vertices.shape[1]


if __name__ == '__main__':
    BU3DFE()[0]
