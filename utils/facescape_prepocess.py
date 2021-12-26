"""Preprocess FaceScape: save original FaceScape datasets as npy file"""

import time
import numpy as np
from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.sphere import Sphere
from tqdm import tqdm

from path import *

if __name__ == '__main__':
    # mesh_viewer = MeshViewers(shape=(1, 1))
    exp_name_list = ["1_neutral",
                     "2_smile",
                     "3_mouth_stretch",
                     "4_anger",
                     "5_jaw_left",
                     "6_jaw_right",
                     "7_jaw_forward",
                     "8_mouth_left",
                     "9_mouth_right",
                     "10_dimpler",
                     "11_chin_raiser",
                     "12_lip_puckerer",
                     "13_lip_funneler",
                     "14_sadness",
                     "15_lip_roll",
                     "16_grin",
                     "17_cheek_blowing",
                     "18_eye_closed",
                     "19_brow_raiser",
                     "20_brow_lower"]
    vertices = []
    subjects = []
    exps = []

    downsample_v_idx = np.load(FACESCAPE_PATH + "downsample/downsampled_v_idx.npy")
    downsample_f = np.load(FACESCAPE_PATH + "downsample/downsampled_f.npy")
    DOWNSAMPLE = True

    for subject in tqdm(range(1, 848)):
        path = FACESCAPE_PATH + str(subject) + "/" + "models_reg/"
        for exp_name in exp_name_list:
            if os.path.exists(path + exp_name + ".obj"):
                mesh = Mesh(filename=path + exp_name + ".obj")
                if DOWNSAMPLE:
                    vertices.append(mesh.v.astype(np.float32)[downsample_v_idx])
                else:
                    vertices.append(mesh.v.astype(np.float32))
                exps.append(exp_name)
                subjects.append(subject)
            else:
                print(path + exp_name + ".obj")

            # Visualisation.
            # for mesh in mesh_list:
            #     mesh_viewer[0][0].set_dynamic_meshes([mesh])

    vertices = np.stack(vertices)
    subjects = np.array(subjects)
    exps = np.array(exps)

    if DOWNSAMPLE:
        np.save(DATASET_PATH + "facescape_faces", downsample_f)
    else:
        np.save(DATASET_PATH + "facescape_faces", mesh.f)
    np.save(DATASET_PATH + "facescape_vertices", vertices)
    np.save(DATASET_PATH + "facescape_subject_ids", subjects)
    np.save(DATASET_PATH + "facescape_expressions", exps)
