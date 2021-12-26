import logging
import sys
from time import time
from types import SimpleNamespace
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from Model.vae import VariationalAE
from path import *
from utils.dataset import BU3DFE
from utils.facescape_dataset import FaceScape

NAME = sys.argv[1]


def face_recog(datasets_type, MODEL_EXPRESSION=0, EXPRESSION_LEVEL=0):
    logging.info(NAME)

    if datasets_type == "BU3DFE":
        test_data = BU3DFE(partition="test", sort=True)
        de_normalise_factor = BU3DFE_NORMALISE
        mean_face = torch.from_numpy(np.load(DATASET_PATH + 'BU3DFE_mean_face.npy')).to(
            device=device, dtype=torch.float32).unsqueeze(0)
        bu3dfe_faces = np.load(DATASET_PATH + "BU3DFE_face.npy")
        vis_face = bu3dfe_faces
    elif datasets_type == "FaceScape":
        test_data = FaceScape(partition='test')
        de_normalise_factor = FACESCAPE_NORMALISE
        mean_face = torch.from_numpy(
            np.load(DATASET_PATH + 'facescape_train_mean_face.npy')).to(device=device,
                                                                             dtype=torch.float32).unsqueeze(
            0)
        facescape_faces = np.load(DATASET_PATH + "facescape_faces.npy")
        vis_face = facescape_faces
    else:
        raise ValueError("No such dataset.")

    # Load prerequisite information.
    with open(LOGS_PATH + NAME + "/config.json", "r") as f:
        args = SimpleNamespace(**json.load(f))
    vae = VariationalAE(args, test_data.vertices_num, args.latent_vector_dim_id,
                        args.latent_vector_dim_exp).to(device)
    vae.load_state_dict(torch.load(LOGS_PATH + NAME + "/VAE-ID_DIS_model_{0}.pt".format(args.epochs)))
    vae.eval()
    test_loader = DataLoader(test_data, num_workers=0, batch_size=1, shuffle=False)

    pred_neu_vertices_list = []
    subject_ids_list = []
    expressions_list = []
    if datasets_type == "BU3DFE":
        exp_level_list = []
    for data in tqdm(test_loader):
        if datasets_type == "BU3DFE":
            true_vertices, expressions, expression_levels, gender, _, subject_ids, _, _, _, _, _, _ = data
            subject_ids = subject_ids.item()
            if gender.item() == 1:
                subject_ids = subject_ids + 100
            expressions = expressions.item()
            expression_levels = expression_levels.item()
            exp_level_list.append(expression_levels)
        elif datasets_type == "FaceScape":
            true_vertices, expressions, subject_ids, _, _, _, _, _, _ = data
            subject_ids = subject_ids.item()
            expressions = expressions.item()
        else:
            raise ValueError("No such dataset.")

        _, pred_vertices_neutral, _, _, _, _, _ = vae(true_vertices)
        pred_vertices_neutral += mean_face
        pred_vertices_neutral = pred_vertices_neutral.squeeze(0).detach().cpu().numpy()
        pred_neu_vertices_list.append(pred_vertices_neutral)
        subject_ids_list.append(subject_ids)
        expressions_list.append(expressions)
    pred_neu_vertices_array = np.stack(pred_neu_vertices_list)
    subject_ids_array = np.array(subject_ids_list)
    expressions_array = np.array(expressions_list)
    if datasets_type == "BU3DFE":
        exp_level_array = np.array(exp_level_list)

    recognition_model_vertices = []
    sid_list = np.unique(subject_ids_array)
    for sid in sid_list:
        if datasets_type == "FaceScape":
            vert = pred_neu_vertices_array[
                np.logical_and(subject_ids_array == sid, expressions_array == MODEL_EXPRESSION)]
        elif datasets_type == "BU3DFE":
            vert = pred_neu_vertices_array[np.logical_and(exp_level_array == EXPRESSION_LEVEL,
                                                          np.logical_and(subject_ids_array == sid,
                                                                         expressions_array == MODEL_EXPRESSION))]
        assert vert.shape[0] == 1
        recognition_model_vertices.append(vert)
    recognition_model_vertices = np.concatenate(recognition_model_vertices)

    recognition_results = []
    for probe_vert, probe_sid, probe_expression in \
            tqdm(zip(pred_neu_vertices_array, subject_ids_array, expressions_array),
                 total=pred_neu_vertices_array.shape[0]):
        if probe_expression == MODEL_EXPRESSION:
            continue
        distances = \
            np.sqrt(np.sum((probe_vert - recognition_model_vertices) ** 2, axis=2)).sum(1)
        pred_sid = sid_list[np.argmin(distances)]
        recognition_results.append(probe_sid == pred_sid)
    print("Recognition Accuracy: %.03f%%" % (sum(recognition_results) / len(recognition_results) * 100.))


if __name__ == '__main__':
    if "BU3DFE" in NAME:
        dataset_type = "BU3DFE"
    elif "FaceScape" in NAME:
        dataset_type = "FaceScape"
    else:
        raise ValueError("Please include dataset name in log name, [COMA|BU3DFE|FaceScape].")
    face_recog(datasets_type=dataset_type, MODEL_EXPRESSION=0, EXPRESSION_LEVEL=0)