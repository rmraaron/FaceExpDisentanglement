"""Preprocess CoMA: save original CoMA datasets as npy file"""

import time
import numpy as np
from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.sphere import Sphere
from tqdm import tqdm

from path import *

if __name__ == '__main__':
    mesh_viewer = MeshViewers(shape=(1, 1))

    subject_list = ["FaceTalk_170725_00137_TA",
                    "FaceTalk_170728_03272_TA",
                    "FaceTalk_170731_00024_TA",
                    "FaceTalk_170809_00138_TA",
                    "FaceTalk_170811_03274_TA",
                    "FaceTalk_170811_03275_TA",
                    "FaceTalk_170904_00128_TA",
                    "FaceTalk_170904_03276_TA",
                    "FaceTalk_170908_03277_TA",
                    "FaceTalk_170912_03278_TA",
                    "FaceTalk_170913_03279_TA",
                    "FaceTalk_170915_00223_TA"]
    exp_name_list = ["bareteeth",
                     "cheeks_in",
                     "eyebrow",
                     "high_smile",
                     "lips_back",
                     "lips_up",
                     "mouth_down",
                     "mouth_extreme",
                     "mouth_middle",
                     "mouth_open",
                     "mouth_side",
                     "mouth_up"]
    # exp_name_list = ["mouth_down",
    #                  "mouth_middle",
    #                  "mouth_side",
    #                  "mouth_up"]
    REF_POINT_IDX = 3506
    vertices = []
    subjects = []
    exps = []
    exp_degree = []

    mesh = None
    for subject in tqdm(subject_list):
        for exp_name in exp_name_list:
            path = COMA_PATH + subject + "/" + exp_name + "/"
            mode = "deviation"
            exp_list = sorted(os.listdir(path))
            neutral_mesh = Mesh(filename=path + exp_name + ".000001.ply")
            neutral_ref_point = neutral_mesh.v[REF_POINT_IDX]
            mesh_list = []
            exp_level_list = []
            for idx, i in enumerate(sorted(os.listdir(path))):
                mesh = Mesh(filename=path + i)

                if mode == "linear":
                    exp_level = idx / (len(exp_list) / 2) if idx <= len(exp_list) / 2 else \
                        (len(exp_list) - idx) / (len(exp_list) / 2)
                elif mode == "deviation":
                    exp_level = np.sqrt(((mesh.v - neutral_mesh.v) ** 2).sum(1)).mean()
                else:
                    raise ValueError()

                mesh_list.append(mesh)
                exp_level_list.append(exp_level)

            # Detecting outliers.
            exp_level_list = np.array(exp_level_list)
            outliers = (exp_level_list - exp_level_list.mean()) / np.std(exp_level_list) > 3
            if outliers.sum() > 0:
                print("Detected outlier:", subject, np.array(os.listdir(path))[outliers])
            mesh_list = np.delete(mesh_list, outliers)
            exp_level_list = np.delete(exp_level_list, outliers)

            exp_level_list = exp_level_list / exp_level_list.max()

            v_list = [m.v.astype(np.float32) for m in mesh_list]
            vertices.append(v_list)
            subjects += [subject] * exp_level_list.shape[0]
            if exp_name in ["mouth_down", "mouth_middle", "mouth_side", "mouth_up"]:
                new_exp_list = []
                for i in range(exp_level_list.shape[0]):
                    if v_list[i][REF_POINT_IDX][0] < neutral_ref_point[0]:
                        new_exp_list.append(exp_name + "_left")
                    else:
                        new_exp_list.append(exp_name + "_right")
            else:
                new_exp_list = [exp_name.split(".")[0]] * exp_level_list.shape[0]
            exps += new_exp_list
            exp_degree.append(exp_level_list)

            # Visualisation.
            # for mesh, new_exp_name, exp_level in zip(mesh_list, new_exp_list, exp_level_list):
            #     vc = np.zeros_like(mesh.v)
            #     if "right" in new_exp_name:
            #         # Right.
            #         vc[:, 1] += exp_level
            #         vc[:, 2] += (1 - exp_level)
            #     else:
            #         # Left.
            #         vc[:, 0] += exp_level
            #         vc[:, 2] += (1 - exp_level)
            #     mesh.vc = vc * 0.9
            #     mesh_viewer[0][0].set_dynamic_meshes([mesh, Sphere(mesh.v[REF_POINT_IDX], 0.002).to_mesh(color=(145, 189, 27))])
            #     time.sleep(0.05)

    vertices = np.concatenate(vertices, axis=0)
    subjects = np.array(subjects)
    exps = np.array(exps)
    exp_degree = np.concatenate(exp_degree)

    np.save(DATASET_PATH + "coma_faces", mesh.f)
    np.save(DATASET_PATH + "coma_vertices", vertices)
    np.save(DATASET_PATH + "coma_subject_ids", subjects)
    np.save(DATASET_PATH + "coma_expressions", exps)
    np.save(DATASET_PATH + "coma_expression_levels", exp_degree.astype(np.float32))
