"""Divide CoMA into training set and test set for DataLoader"""

import logging
import time

import torch
from psbody.mesh import MeshViewers, Mesh
from torch.utils.data import Dataset

import numpy as np
import platform
from path import *


SUBJECT_ID_MAPPING = {"FaceTalk_170725_00137_TA": 0,
                      "FaceTalk_170728_03272_TA": 1,
                      "FaceTalk_170731_00024_TA": 2,
                      "FaceTalk_170809_00138_TA": 3,
                      "FaceTalk_170811_03274_TA": 4,
                      "FaceTalk_170811_03275_TA": 5,
                      "FaceTalk_170904_00128_TA": 6,
                      "FaceTalk_170904_03276_TA": 7,
                      "FaceTalk_170908_03277_TA": 8,
                      "FaceTalk_170912_03278_TA": 9,
                      "FaceTalk_170913_03279_TA": 10,
                      "FaceTalk_170915_00223_TA": 11}
EXPRESSION_MAPPING = {"bareteeth": 0, "cheeks_in": 1, "eyebrow": 2, "high_smile": 3, "lips_back": 4,
                      "lips_up": 5, "mouth_down_left": 6, "mouth_extreme": 7, "mouth_middle_left": 8, "mouth_open": 9,
                      "mouth_side_left": 10, "mouth_up_left": 11, "mouth_down_right": 12, "mouth_middle_right": 13,
                      "mouth_side_right": 14, "mouth_up_right": 15}


class COMA(Dataset):
    def __init__(self, partition="train", split_scheme="fixed", always_sample_same_id=False):
        self.partition = partition
        self.neutral_threshold = 0.01
        vertices = np.load(DATASET_PATH + "coma_vertices.npy")
        subject_ids = np.load(DATASET_PATH + "coma_subject_ids.npy")
        expressions = np.load(DATASET_PATH + "coma_expressions.npy")
        expression_levels = np.load(DATASET_PATH + "coma_expression_levels.npy")

        subject_ids = np.vectorize(SUBJECT_ID_MAPPING.get)(subject_ids)
        expressions = np.vectorize(EXPRESSION_MAPPING.get)(expressions)

        # Normalisation.
        vertices /= COMA_NORMALISE

        # Split train and test.
        if os.path.exists(DATASET_PATH + "coma_split_scheme_" + split_scheme + ".npy"):
            split_scheme_idx = np.load(DATASET_PATH + "coma_split_scheme_" + split_scheme + ".npy")
        else:
            # Split by 10% of the dataset. Sample sequences of length 10.
            num_seq = int(vertices.shape[0] * 0.1 * 0.1)
            split_scheme_idx = []
            for i in range(num_seq):
                valid_sample = False
                while not valid_sample:
                    random_idx = np.random.randint(0, vertices.shape[0] - 10)
                    if subject_ids[random_idx] == subject_ids[random_idx + 9] and \
                            expressions[random_idx] == expressions[random_idx + 9] and \
                            sum([j - 10 < random_idx < j + 10 for j in split_scheme_idx]) == 0:
                        split_scheme_idx.append(random_idx)
                        valid_sample = True
            split_scheme_idx = np.array(sorted(split_scheme_idx))
            np.save(DATASET_PATH + "coma_split_scheme_" + split_scheme, split_scheme_idx)

        split_scheme_idx = np.concatenate([split_scheme_idx + i for i in range(10)])
        test_partition_idx = np.zeros_like(subject_ids).astype(np.bool)
        test_partition_idx[split_scheme_idx] = True

        if partition == "test":
            all_neutral_idx = expression_levels <= self.neutral_threshold
            neutral_vertices = vertices[all_neutral_idx]
            neutral_subject_ids = subject_ids[all_neutral_idx]

            vertices = vertices[test_partition_idx]
            subject_ids = subject_ids[test_partition_idx]
            expressions = expressions[test_partition_idx]
            expression_levels = expression_levels[test_partition_idx]
        else:
            neutral_vertices = None
            neutral_subject_ids = None
            vertices = vertices[np.logical_not(test_partition_idx)]
            subject_ids = subject_ids[np.logical_not(test_partition_idx)]
            expressions = expressions[np.logical_not(test_partition_idx)]
            expression_levels = expression_levels[np.logical_not(test_partition_idx)]

        if partition == "train":
            # Save average normalised point cloud.
            average_face = np.average(vertices, axis=0)
            np.save(DATASET_PATH + "coma_average_vertices_" + split_scheme, average_face)
            if os.path.exists(DATASET_PATH + "coma_exp_gt_dict.pt"):
                logging.info("Using existing exp gt dict.")
                self.expression_gt_dict = torch.load(DATASET_PATH + "coma_exp_gt_dict.pt")
            else:
                self.expression_gt_dict = {}
                # mesh_viewer = MeshViewers(shape=(1, 1))
                for i in range(16):
                    for j_b, j_u in [(0., 0.3), (0.2, 0.8), (0.7, 1.0)]:
                        if i not in self.expression_gt_dict:
                            self.expression_gt_dict[i] = {}
                        exp_average_gt = \
                            np.average(vertices[np.logical_and(expressions == i,
                                                               np.logical_and(expression_levels >= j_b,
                                                                              expression_levels <= j_u))], axis=0)
                        self.expression_gt_dict[i][j_b] = torch.from_numpy(exp_average_gt)
                        # coma_faces = np.load(DATASET_PATH + "coma_faces.npy")
                        # mesh_exp_true = Mesh(f=coma_faces)
                        # mesh_exp_true.v = exp_average_gt
                        # mesh_viewer[0][0].set_dynamic_meshes([mesh_exp_true])
                        # time.sleep(1)
                torch.save(self.expression_gt_dict, DATASET_PATH + "coma_exp_gt_dict.pt")
        else:
            self.expression_gt_dict = torch.load(DATASET_PATH + "coma_exp_gt_dict.pt")

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
            self.vertices = torch.from_numpy(vertices).to(device)
            self.expressions = torch.from_numpy(expressions).to(dtype=torch.long, device=device)
            self.expression_levels = torch.from_numpy(expression_levels).to(device=device)
            self.subject_ids = torch.from_numpy(subject_ids).to(dtype=torch.long, device=device)
            if neutral_vertices is not None:
                self.neutral_vertices = torch.from_numpy(neutral_vertices).to(device)
                self.neutral_subject_ids = torch.from_numpy(neutral_subject_ids).to(device)
            self.always_sample_same_id = always_sample_same_id
            self.exp_gt_dict = move_to(self.expression_gt_dict, device)
            self.expression_gt_dict = self.exp_gt_dict
        else:
            self.vertices = torch.from_numpy(vertices)
            self.expressions = torch.from_numpy(expressions).to(dtype=torch.long, device=device)
            self.expression_levels = torch.from_numpy(expression_levels).to(device=device)
            self.subject_ids = torch.from_numpy(subject_ids).to(dtype=torch.long, device=device)
            if neutral_vertices is not None:
                self.neutral_vertices = torch.from_numpy(neutral_vertices).to(device)
                self.neutral_subject_ids = torch.from_numpy(neutral_subject_ids).to(device)
            self.always_sample_same_id = always_sample_same_id

    def __getitem__(self, index):
        """
        Return: Input vertices;
                Expression ~[0, 11];
                Expression level ~[0., 1.];
                Subject ID ~[0, 11];
                Corresponding neutral vertices;
                Sampled ID vertices which has;
                Same ID or not;
                Sampled Exp vertices which has;
                Same Exp or not;
        """
        if self.partition == "train":
            neutral_v = self.vertices[self.get_neutral_idx(self.subject_ids[index])].mean(0)
        elif self.partition == "test":
            neutral_v = self.neutral_vertices[self.get_neutral_idx(self.subject_ids[index])].mean(0)
        else:
            raise ValueError

        if not self.always_sample_same_id:
            id_label_same = torch.randint(2, (1,)) == 1
        else:
            id_label_same = torch.tensor([True], dtype=torch.bool)

        exp_label_same = torch.randint(2, (1,)) == 1

        sample_id_idx = self.sample_id(self.subject_ids[index], id_label_same.item())
        sample_exp_idx = self.sample_exp(self.expressions[index], exp_label_same.item())

        exp_lvl = self.expression_levels[index].item()
        if 0. <= exp_lvl <= 0.375:
            exp_lvl_key = 0.
        elif 0.375 < exp_lvl <= 0.75:
            exp_lvl_key = 0.2
        elif 0.75 < exp_lvl <= 1.0:
            exp_lvl_key = 0.7
        else:
            raise ValueError
        exp_v = self.expression_gt_dict[self.expressions[index].item()][exp_lvl_key]

        if platform.system() == "Linux":
            return self.vertices[index], \
                   self.expressions[index], \
                   self.expression_levels[index], \
                   self.subject_ids[index], \
                   neutral_v, \
                   exp_v, \
                   self.vertices[sample_id_idx], \
                   id_label_same.to(device, dtype=torch.long).squeeze(-1), \
                   self.vertices[sample_exp_idx].to(device), \
                   exp_label_same.to(device, dtype=torch.long).squeeze(-1),
        else:
            return self.vertices[index].to(device), \
                self.expressions[index], \
                self.expression_levels[index], \
                self.subject_ids[index], \
                neutral_v.to(device), \
                exp_v.to(device), \
                self.vertices[sample_id_idx].to(device), \
                id_label_same.to(device, dtype=torch.long).squeeze(-1), \
                self.vertices[sample_exp_idx].to(device), \
                exp_label_same.to(device, dtype=torch.long).squeeze(-1),

    def __len__(self):
        return self.vertices.shape[0]

    def get_neutral_idx(self, id_):
        if self.partition == "train":
            # Expression level threshold set to 0.05. Since no neutral in COMA dataset.
            neutral_idx = torch.nonzero(torch.logical_and(self.subject_ids == id_,
                                                          self.expression_levels <= self.neutral_threshold))
        elif self.partition == "test":
            neutral_idx = torch.nonzero(self.neutral_subject_ids == id_)
        else:
            raise ValueError
        # return neutral_idx[torch.randint(neutral_idx.size(0), (1,)), 0].item()
        return neutral_idx.flatten()

    def sample_id(self, target, same):
        if same:
            id_idx = torch.nonzero(self.subject_ids == target)
        else:
            id_idx = torch.nonzero(self.subject_ids != target)

        rand_idx = torch.randint(id_idx.shape[0], (1,))

        return id_idx[rand_idx].item()

    def sample_exp(self, target, same):
        if same:
            exp_idx = torch.nonzero(self.expressions == target)  # BUDUI
        else:
            exp_idx = torch.nonzero(self.expressions != target)  # BUDUI

        rand_idx = torch.randint(exp_idx.shape[0], (1,))

        return exp_idx[rand_idx].item()

    @property
    def vertices_num(self):
        return self.vertices.shape[1]

    @property
    def id_label_num(self):
        return 0

    @property
    def expression_num(self):
        return 12


if __name__ == '__main__':
    a = COMA("train")[10]
