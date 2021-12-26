"""Preprocess BU3DFE: save original BU3DFE datasets as npy file"""


import os
import pickle
import numpy as np
import re
from tqdm import tqdm
from path import *


def open_wrlfile(wrl_filename):
    with open(BU3DFE_PATH + wrl_filename, 'r') as vrml:
        i = 0
        j = 0
        # points_list is used to save all points coordinates, each point is represented as one element
        points_list = []
        face_list = []
        for lines in vrml:
            if i > 6 and i < 6003:
                line = lines.split('\n')[0]
                points = line.split(',')[0]
                # point_list is used to save one point, each element represent one axis (x, y, z)
                point_list = []
                point_list.append(float(points.split()[0]))
                point_list.append(float(points.split()[1]))
                point_list.append(float(points.split()[2]))
                points_list.append(point_list)
            i += 1

            if j > 6005 and j < 17759:
                line = lines.split('\n')[0]
                coordinates = line.split('-1')[0]
                coor_list = []
                coor_list.append(int(coordinates.split(', ')[0]))
                coor_list.append(int(coordinates.split(', ')[1]))
                coor_list.append(int(coordinates.split(', ')[2]))
                face_list.append(coor_list)
            j += 1
        points_array = np.array(points_list, dtype=np.float)
        face_array = np.array(face_list)
        return points_array, face_array


def compute_rt(probe_pts, model_pts):
    if probe_pts.shape[0] != model_pts.shape[0] or probe_pts.shape[1] != model_pts.shape[1]:
        raise ValueError("Probe and model have different numbers of points.")
    if probe_pts.shape[1] != 2 and probe_pts.shape[1] != 3:
        raise ValueError("Probe and model have wrong number of dimensions (only 2 or 3 allowed).")

    probe_mean = np.mean(probe_pts, axis=0)
    probe_pts_zm = probe_pts - probe_mean
    model_mean = np.mean(model_pts, axis=0)
    model_pts_zm = model_pts - model_mean

    B = probe_pts_zm.T @ model_pts_zm
    U, _, VH = np.linalg.svd(B)
    V = VH.T
    R = V @ U.T

    if np.linalg.det(R) < 0:
        if probe_pts.shape[1] == 3:
            R = V @ np.diag([1, 1, -1]) @ U.T
        else:
            R = V @ np.diag([1, -1]) @ U.T

    T = model_mean - R @ probe_mean

    return R, T


BU3DFE_NR_LANDMARKS = [4302, 954, 227, 1700]
BU3DFE_NR_SUPP_LANDMARKS = [2186, 1128, 4303, 1597, 4501]
BU3DFE_EC_LANDMARKS = [4827, 41, 3604, 1143]
BU3DFE_INVAR_LANDMARKS = BU3DFE_NR_LANDMARKS + BU3DFE_NR_SUPP_LANDMARKS + BU3DFE_EC_LANDMARKS


def bu3dfe_normalise():
    train_set = np.load(DATASET_PATH + "BU3DFE_ver_train.npy")
    train_set_labels = np.load(DATASET_PATH + "BU3DFE_label_train.npy", allow_pickle=True)[()]
    test_set = np.load(DATASET_PATH + "BU3DFE_ver_test.npy")
    test_set_labels = np.load(DATASET_PATH + "BU3DFE_label_test.npy", allow_pickle=True)[()]

    average_neutral_train = train_set[np.array(train_set_labels["expression"]) == "NE"].mean(0)

    def process_set(data, labels):
        genders = np.array(labels["subject_gender"])
        subject_ids = np.array(labels["subject_id"])
        expressions = np.array(labels["expression"])

        # Process neutrals.
        new_ne = []
        for v, sid, g in tqdm(zip(data[expressions == "NE"],
                                  subject_ids[expressions == "NE"],
                                  genders[expressions == "NE"]),
                              total=data[expressions == "NE"].shape[0]):
            R, T = compute_rt(v, average_neutral_train)  # Compute using all vertices.
            new_v = v @ R.T + T
            new_ne.append(new_v)

            # mesh.v = new_v
            # mesh_viewer[0][0].set_dynamic_meshes([mesh])
            # sleep(0.01)
        new_ne = np.stack(new_ne)
        data[expressions == "NE"] = np.stack(new_ne)

        # Process expressions.
        for i in tqdm(range(data.shape[0])):
            v = data[i]
            gender = genders[i]
            sid = subject_ids[i]
            exp = expressions[i]

            if exp == "NE":
                continue

            v_ne = data[np.logical_and.reduce(np.stack([genders == gender,
                                                        subject_ids == sid,
                                                        expressions == "NE"]))]
            assert v_ne.shape[0] == 1  # No two neutrals for the same identity.
            v_ne = v_ne[0]

            R, T = compute_rt(v[BU3DFE_INVAR_LANDMARKS], v_ne[BU3DFE_INVAR_LANDMARKS])
            data[i] = v @ R.T + T

        return data
    new_train_set = process_set(train_set, train_set_labels)
    new_test_set = process_set(test_set, test_set_labels)
    np.save(DATASET_PATH + "BU3DFE_ver_train_reg", new_train_set)
    np.save(DATASET_PATH + "BU3DFE_ver_test_reg", new_test_set)


if __name__ == '__main__':
    filename_pattern = re.compile('^[FM][0-9]{4}_.{6}_F3D.wrl')
    file_list = []
    for fn in os.listdir(BU3DFE_PATH):
        if re.match(filename_pattern, fn):
            file_list.append(fn)

    FEMALE_TEST_SET, MALE_TEST_SET = [1, 2, 3, 4, 5, 6], [1, 2, 3, 5]
    test_set_starts = ['F%04d' % i for i in FEMALE_TEST_SET] + ['M%04d' % i for i in MALE_TEST_SET]

    vertices_list = [[], []]
    labels_list = []

    for i in range(2):
        labels_list.append({'subject_id': [], 'subject_gender': [],
                            'expression': [], 'expression_level': [], 'race': []})

    for fn in tqdm(file_list):
        vertices, face = open_wrlfile(fn)
        if sum([j in fn for j in test_set_starts]) > 0:
            # Test set.
            dataset_id = 1
        else:
            dataset_id = 0
        vertices_list[dataset_id].append(vertices)
        labels_list[dataset_id]['subject_id'].append(int(fn[1:5]))
        labels_list[dataset_id]['subject_gender'].append(fn[0])
        labels_list[dataset_id]['expression'].append(fn[6:8])
        labels_list[dataset_id]['expression_level'].append(int(fn[8:10]))
        labels_list[dataset_id]['race'].append(fn[10:12])

    np.save(DATASET_PATH + 'BU3DFE_ver_train', np.stack(vertices_list[0]))
    np.save(DATASET_PATH + 'BU3DFE_ver_test', np.stack(vertices_list[1]))
    np.save(DATASET_PATH + 'BU3DFE_label_train', labels_list[0])
    np.save(DATASET_PATH + 'BU3DFE_label_test', labels_list[1])
    np.save(DATASET_PATH + 'BU3DFE_face', face)

    bu3dfe_normalise()






