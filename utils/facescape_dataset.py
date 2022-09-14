"""Divide FaceScape into training set and test set for DataLoader"""


import logging
import torch
from torch.utils.data import Dataset
import numpy as np
from path import *
from psbody.mesh import MeshViewers, Mesh
import time
import random
import platform
# import msvcrt as m

EXPRESSION_MAPPING = {"1_neutral": 0, "2_smile": 1, "3_mouth_stretch": 2, "4_anger": 3, "5_jaw_left": 4,
                      "6_jaw_right": 5, "7_jaw_forward": 6, "8_mouth_left": 7, "9_mouth_right": 8, "10_dimpler": 9,
                      "11_chin_raiser": 10, "12_lip_puckerer": 11, "13_lip_funneler": 12, "14_sadness": 13,
                      "15_lip_roll": 14, "16_grin": 15, "17_cheek_blowing": 16, "18_eye_closed": 17, "19_brow_raiser": 18,
                      "20_brow_lower": 19}


class FaceScape(Dataset):
    def __init__(self, partition="train", include_neutral=True, always_sample_same_id=False):
        vertices = np.load(DATASET_PATH + "facescape_vertices.npy")
        subject_ids = np.load(DATASET_PATH + "facescape_subject_ids.npy")
        expressions = np.load(DATASET_PATH + "facescape_expressions.npy")

        expressions = np.vectorize(EXPRESSION_MAPPING.get)(expressions)

        # check outliers

        # avg_dist = np.sqrt(np.sum((vertices - vertices.mean(0)[None, :, :]) ** 2,
        #                      axis=2)).mean(1)
        # outliers = np.where(np.abs((avg_dist - avg_dist.mean()) / avg_dist.std()) > 3)
        # mesh_viewer = MeshViewers(shape=(1, 1))
        # facescape_faces = np.load(DATASET_PATH + "facescape_faces.npy")
        # for i in outliers[0]:
        #     mesh_viewer[0][0].set_dynamic_meshes([Mesh(v=vertices[i], f=facescape_faces)])
        #     print(i, expressions[i], subject_ids[i])
        #     while mesh_viewer[0][0].get_keypress() != b"n":
        #         pass
        # subject_id: 603
        # for i in range(12022, 12042):
        #     mesh_viewer[0][0].set_dynamic_meshes([Mesh(v=vertices[i], f=facescape_faces)])
        #     print(expressions[i], subject_ids[i])
        #     while mesh_viewer[0][0].get_keypress() != b"n":
        #         pass
        # quit()

        outliers_list = [451, 1313, 4179, 6399, 10416, 12022, 12023, 12024, 12025, 12026, 12027,
                         12028, 12029, 12030, 12031, 12032, 12033, 12034, 12035, 12036, 12037, 12038,
                         12039, 12040, 12041, 12824, 12894, 12897, 12898, 13992, 14584, 14624, 16621]
        mask = np.ones(shape=(vertices.shape[0],), dtype=np.bool)
        mask[outliers_list] = False
        vertices = vertices[mask]
        subject_ids = subject_ids[mask]
        expressions = expressions[mask]

        if not os.path.exists(DATASET_PATH + "facescape_each_id_mean_face.npy"):
            # total number of subject_ids
            id_num_sum = subject_ids[-1]
            id_mean_face = {}
            for i in range(1, id_num_sum+1):
                vertex_indices = np.where(subject_ids == i)
                each_id_vertices = vertices[vertex_indices[0]]
                each_id_mean_face = np.average(each_id_vertices, axis=0)
                id_mean_face[i] = each_id_mean_face
            np.save(DATASET_PATH + "facescape_each_id_mean_face", id_mean_face)

        # Normalisation.
        vertices /= FACESCAPE_NORMALISE

        # 30% test data ids
        random.seed(50)
        # The dataset providers list the exact ids for publications
        publicable_list = [122, 212, 340, 344, 393, 395, 421, 527, 594, 610]
        test_ids_list = random.sample(set(subject_ids), int(len(set(subject_ids)) * 0.3))
        k = 0
        for publicable_id in publicable_list:
            if publicable_id not in test_ids_list:
                test_ids_list[k] = np.int32(publicable_id)
                k += 1
        test_ids_list.sort()
        # test ids index
        subject_indices_list = []
        for test_id in test_ids_list:
            subject_indices = np.where(subject_ids == test_id)[0]
            for subject_idx in subject_indices:
                subject_indices_list.append(subject_idx)

        if partition == "test":
            vertices = vertices[subject_indices_list]
            subject_ids = subject_ids[subject_indices_list]
            expressions = expressions[subject_indices_list]
        else:
            train_indices_list = []
            for idx in range(len(subject_ids)):
                if idx not in subject_indices_list:
                    train_indices_list.append(idx)
            vertices = vertices[train_indices_list]
            subject_ids = subject_ids[train_indices_list]
            expressions = expressions[train_indices_list]

        if partition == "train":
            # Save average normalised point cloud.
            mean_face = np.average(vertices, axis=0)
            np.save(DATASET_PATH + "facescape_" + partition + "_mean_face_70percent", mean_face)
            if os.path.exists(DATASET_PATH + "facescape_exp_gt_dict_70percent.pt"):
                logging.info("Using existing exp gt dict.")
                self.expression_gt_dict = torch.load(DATASET_PATH + "facescape_exp_gt_dict_70percent.pt")
            else:
                self.expression_gt_dict = {}
                mesh_viewer = MeshViewers(shape=(1, 2))

                # i: 20 expressions including neutral
                for i in range(20):
                    exp_average_gt = np.average(vertices[expressions == i], axis=0)
                    self.expression_gt_dict[i] = torch.from_numpy(exp_average_gt).to(torch.float32)
                    facescape_faces = np.load(DATASET_PATH + "facescape_faces.npy")
                    mesh_exp_true = Mesh(f=facescape_faces)
                    mesh_exp_true.v = exp_average_gt
                    mesh_mean_face = Mesh(f=facescape_faces)
                    mesh_mean_face.v = mean_face
                    mesh_viewer[0][0].set_dynamic_meshes([mesh_exp_true])
                    mesh_viewer[0][1].set_dynamic_meshes([mesh_mean_face])
                    time.sleep(1)
                torch.save(self.expression_gt_dict, DATASET_PATH + "facescape_exp_gt_dict_70percent.pt")
        else:
            self.expression_gt_dict = torch.load(DATASET_PATH + "facescape_exp_gt_dict_70percent.pt")

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
            self.subject_ids = torch.from_numpy(subject_ids).to(device)
            self.exp_gt_dict = move_to(self.expression_gt_dict, device)
            self.expression_gt_dict = self.exp_gt_dict
        else:
            self.vertices = torch.from_numpy(vertices).to(torch.float32)
            self.expressions = torch.from_numpy(expressions)
            self.subject_ids = torch.from_numpy(subject_ids)

        if not include_neutral:
            self.vertices = self.vertices[expressions != 0]
            self.expressions = self.expressions[expressions != 0] - 1
            self.subject_ids = self.subject_ids[expressions != 0]

        self.always_sample_same_id = always_sample_same_id

    def __len__(self):
        return self.vertices.shape[0]

    def __getitem__(self, index):
        if not self.always_sample_same_id:
            same_id_label = torch.randint(2, (1,)) == 1
        else:
            same_id_label = torch.tensor([True], dtype=torch.bool)
        sample_id_idx = self.sample_id(self.subject_ids[index], same_id_label.item())
        same_exp_label = torch.randint(2, (1,)) == 1
        sample_exp_idx = self.sample_exp(self.expressions[index], same_exp_label.item())
        neutral_id_idx = self.get_neutral(self.subject_ids[index])
        exp_v = self.expression_gt_dict[self.expressions[index].item()]

        if platform.system() == "Linux":
            return self.vertices[index], \
                   self.expressions[index], \
                   self.subject_ids[index], \
                   self.vertices[sample_id_idx], \
                   same_id_label.to(device, dtype=torch.long).squeeze(-1), \
                   self.vertices[sample_exp_idx], \
                   same_exp_label.to(device, dtype=torch.long).squeeze(-1), \
                   self.vertices[neutral_id_idx], \
                   exp_v
        else:
            return self.vertices[index].to(device), \
                   self.expressions[index].to(device), \
                   self.subject_ids[index].to(device), \
                   self.vertices[sample_id_idx].to(device), \
                   same_id_label.to(device, dtype=torch.long).squeeze(-1), \
                   self.vertices[sample_exp_idx].to(device), \
                   same_exp_label.to(device, dtype=torch.long).squeeze(-1), \
                   self.vertices[neutral_id_idx].to(device), \
                   exp_v.to(device)

    def sample_id(self, target, same):
        if same:
            id_idx = torch.nonzero(self.subject_ids == target)
        else:
            id_idx = torch.nonzero(self.subject_ids != target)

        rand_idx = torch.randint(id_idx.shape[0], (1,))

        return id_idx[rand_idx].item()

    def sample_exp(self, target, same):
        if same:
            exp_idx = torch.nonzero(self.expressions == target)
        else:
            exp_idx = torch.nonzero(self.expressions != target)

        rand_idx = torch.randint(exp_idx.shape[0], (1,))

        return exp_idx[rand_idx].item()

    def get_neutral(self, id_):
        id_idx = torch.nonzero(torch.logical_and(self.subject_ids == id_, self.expressions == 0))
        return id_idx.item()

    @property
    def vertices_num(self):
        return self.vertices.shape[1]

    @property
    def expression_num(self):
        return 20


if __name__ == '__main__':
    train_data = FaceScape(partition='train')
