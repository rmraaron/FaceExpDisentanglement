import json
import logging
import sys
import time
from types import SimpleNamespace

from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from psbody.mesh import MeshViewers, Mesh
import trimesh

from Model.vae import VariationalAE
from utils.coma_dataset import COMA
from utils.dataset import BU3DFE
from utils.facescape_dataset import FaceScape
from path import *

random_subject_COMA_a = 7
random_subject_COMA_b = 10
random_subject_a_FaceScape = 212
random_subject_b_FaceScape = 594

random_COMA_interpolate_a = 8
random_COMA_interpolate_b = 9
random_FaceScape_interpolate_b = 610
random_FaceScape_interpolate_a = 594

NAME = sys.argv[1]


def visualise(vis_face, vertices_list, datasets_type, show_type):
    # Setup for mesh visualisation.
    if show_type == "exp_transfer":
        if not os.path.exists(LOGS_PATH + "exp_transfer/"):
            os.mkdir(LOGS_PATH + "exp_transfer/")
        mesh_viewer = MeshViewers(shape=(2, 2))
        mesh_viewer[0][0].set_background_color(np.array([1., 1., 1.]))

        mesh_vis_true_a = Mesh(f=vis_face)
        mesh_vis_true_b = Mesh(f=vis_face)
        mesh_vis_transfer_a = Mesh(f=vis_face)
        mesh_vis_transfer_b = Mesh(f=vis_face)
        vertices_a_gt = vertices_list[0]
        vertices_b_gt = vertices_list[1]
        if not os.path.exists(LOGS_PATH + "exp_transfer/" + "{}/".format(datasets_type)):
            os.mkdir(LOGS_PATH + "exp_transfer/" + "{}/".format(datasets_type))
        if datasets_type == "COMA":
            random_a = random_subject_COMA_a
            random_b = random_subject_COMA_b
        elif datasets_type == "FaceScape":
            random_a = random_subject_a_FaceScape
            random_b = random_subject_b_FaceScape

        for i, [vertices_a_transfer, vertices_b_transfer] in enumerate(
                zip(vertices_list[2::2], vertices_list[3::2])):
            mesh_vis_true_a.v = vertices_a_gt
            mesh_vis_true_b.v = vertices_b_gt
            mesh_vis_transfer_a.v = vertices_a_transfer
            mesh_vis_transfer_b.v = vertices_b_transfer

            mesh_viewer[0][0].set_dynamic_meshes([mesh_vis_transfer_a], blocking=True)
            mesh_viewer[0][1].set_dynamic_meshes([mesh_vis_transfer_b], blocking=True)
            mesh_viewer[1][0].set_dynamic_meshes([mesh_vis_true_a], blocking=True)
            mesh_viewer[1][1].set_dynamic_meshes([mesh_vis_true_b], blocking=True)
            mesh_viewer[1][1].save_snapshot(
                LOGS_PATH + "exp_transfer/" + "{}/".format(datasets_type)
                + "{0}_{1}_{2}.png".format(random_a, random_b, i))
            time.sleep(0.1)

        save_mesh = trimesh.Trimesh(vertices=vertices_a_gt, faces=vis_face)
        save_mesh.export(
            LOGS_PATH + "exp_transfer/" + "{}/".format(datasets_type) + "gt_{}.ply".format(
                random_a), "ply")
        save_mesh = trimesh.Trimesh(vertices=vertices_b_gt, faces=vis_face)
        save_mesh.export(
            LOGS_PATH + "exp_transfer/" + "{}/".format(datasets_type) + "gt_{}.ply".format(
                random_b), "ply")
        save_mesh = trimesh.Trimesh(vertices=vertices_list[-2], faces=vis_face)
        save_mesh.export(
            LOGS_PATH + "exp_transfer/" + "{}/".format(datasets_type) + "transferred_{}.ply".format(
                random_a), "ply")
        save_mesh = trimesh.Trimesh(vertices=vertices_list[-1], faces=vis_face)
        save_mesh.export(
            LOGS_PATH + "exp_transfer/" + "{}/".format(datasets_type) + "transferred_{}.ply".format(
                random_b), "ply")

    elif show_type == "exp_interpolate":
        if not os.path.exists(LOGS_PATH + "exp_interpolate/"):
            os.mkdir(LOGS_PATH + "exp_interpolate/")
        mesh_viewer = MeshViewers(shape=(1, 1))
        mesh_viewer[0][0].set_background_color(np.array([1., 1., 1.]))
        if not os.path.exists(LOGS_PATH + "exp_interpolate/" + "{}/".format(datasets_type)):
            os.mkdir(LOGS_PATH + "exp_interpolate/" + "{}/".format(datasets_type))
        if datasets_type == "COMA":
            random_a = random_COMA_interpolate_a
            random_b = random_COMA_interpolate_b
        elif datasets_type == "FaceScape":
            random_a = random_FaceScape_interpolate_a
            random_b = random_FaceScape_interpolate_b
        for i, ver in enumerate(vertices_list):
            mesh_vis = Mesh(f=vis_face)
            mesh_vis.v = ver
            mesh_viewer[0][0].set_dynamic_meshes([mesh_vis], blocking=True)
            mesh_viewer[0][0].save_snapshot(LOGS_PATH + "exp_interpolate/" + "{}/".format(
                datasets_type) + "id_{0}_{1}_{2}.png".format(random_a, random_b, i))
            time.sleep(0.1)
            save_mesh = trimesh.Trimesh(vertices=ver, faces=vis_face)
            save_mesh.export(
                LOGS_PATH + "exp_interpolate/" + "{}/".format(datasets_type) + "id_{}.ply".format(
                    i), "ply")


def load_model(show_type, datasets_type, with_gt):
    logging.info(NAME)
    # Load data.
    if datasets_type == "COMA":
        test_data = COMA(partition='test')
        de_normalise_factor = COMA_NORMALISE * 1000
        mean_face = torch.from_numpy(
            np.load(DATASET_PATH + 'coma_average_vertices_fixed.npy')).to(device=device,
                                                                          dtype=torch.float32).unsqueeze(
            0)
        coma_faces = np.load(DATASET_PATH + "coma_faces.npy")
        vis_face = coma_faces
    elif datasets_type == "BU3DFE":
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

    # Load trained model.
    vae = VariationalAE(args, test_data.vertices_num, args.latent_vector_dim_id,
                            args.latent_vector_dim_exp).to(device)

    vae.load_state_dict(torch.load(LOGS_PATH + NAME + "/VAE-ID_DIS_model_{0}.pt".format(args.epochs)))

    vae.eval()

    two_persons_vertices = {}
    two_persons_exps = {}

    for i, data in enumerate(test_data):
        # Obtain data.
        if datasets_type == "COMA":
            true_vertices, expressions, expression_levels, subject_ids, true_neutral_vertices, _, _, _, _, _ = data
            subject_ids = subject_ids.item()
            expressions = expressions.item()
            expression_levels = expression_levels.item()
            if show_type == "exp_transfer":
                if subject_ids == random_subject_COMA_a and expressions == 6 and len(
                        two_persons_vertices) == 0:
                    two_persons_vertices[subject_ids] = true_vertices
                    two_persons_exps[subject_ids] = expressions
                else:
                    if len(two_persons_vertices) == 1:
                        if subject_ids == random_subject_COMA_b and expressions != \
                                list(two_persons_exps.values())[0] and expressions != 0:
                            two_persons_vertices[subject_ids] = true_vertices
                            two_persons_exps[subject_ids] = expressions
            elif show_type == "exp_interpolate":
                if with_gt:
                    if subject_ids == random_COMA_interpolate_a and expression_levels > 0.9 and len(two_persons_vertices) < 2:
                        two_persons_vertices[subject_ids] = true_neutral_vertices
                        two_persons_vertices[expression_levels] = true_vertices
                        two_persons_exps[subject_ids] = subject_ids
                        two_persons_exps[expression_levels] = expressions
                    else:
                        if len(two_persons_vertices) == 2 and subject_ids == random_COMA_interpolate_b and expression_levels > 0.9:
                            two_persons_vertices[subject_ids] = true_neutral_vertices
                            two_persons_vertices[expression_levels] = true_vertices
                            two_persons_exps[subject_ids] = subject_ids
                            two_persons_exps[expression_levels] = expressions

        elif datasets_type == "BU3DFE":
            true_vertices, expressions, _, _, _, subject_ids, _, _, _, _, _, _ = data
            subject_ids = subject_ids.item()
            expressions = expressions.item()
        elif datasets_type == "FaceScape":
            true_vertices, expressions, subject_ids, _, _, _, _, true_neutral_vertices, _ = data
            subject_ids = subject_ids.item()
            expressions = expressions.item()
            if show_type == "exp_transfer":
                if subject_ids == random_subject_a_FaceScape and expressions == 14 and len(
                        two_persons_vertices) == 0:
                    two_persons_vertices[subject_ids] = true_vertices
                    two_persons_exps[subject_ids] = expressions
                else:
                    if len(two_persons_vertices) == 1:
                        if subject_ids == random_subject_b_FaceScape and expressions == 2:
                            two_persons_vertices[subject_ids] = true_vertices
                            two_persons_exps[subject_ids] = expressions
            elif show_type == "exp_interpolate":
                if with_gt:
                    if subject_ids == random_FaceScape_interpolate_a and expressions == 2 and len(two_persons_vertices) < 2:
                        two_persons_vertices[subject_ids] = true_neutral_vertices
                        two_persons_vertices[0] = true_vertices
                    if subject_ids == random_FaceScape_interpolate_b and expressions == 2 and len(two_persons_vertices) == 2:
                        two_persons_vertices[subject_ids] = true_neutral_vertices
                        two_persons_vertices[expressions] = true_vertices
        else:
            raise ValueError("No such dataset.")

    return vis_face, two_persons_vertices, de_normalise_factor, mean_face, vae


def expression_transfer(datasets_type="COMA", with_gt=True):
    vis_face, two_persons_vertices, de_normalise_factor, mean_face, vae = load_model(show_type="exp_transfer", datasets_type=datasets_type, with_gt=with_gt)
    true_vertices_a = list(two_persons_vertices.values())[0].unsqueeze(0)
    true_vertices_b = list(two_persons_vertices.values())[1].unsqueeze(0)
    z_id_a, z_exp_a = vae.encoding_eval(true_vertices_a)
    z_id_b, z_exp_b = vae.encoding_eval(true_vertices_b)

    vertices_a_gt = true_vertices_a[0].detach().cpu().numpy() * de_normalise_factor
    vertices_b_gt = true_vertices_b[0].detach().cpu().numpy() * de_normalise_factor
    vertices_list = [vertices_a_gt, vertices_b_gt]
    for i in range(1, 151):
        transfer_vertices_a, _, _ = vae.decoding(z_id_a, z_exp_a + (z_exp_b - z_exp_a) / 150 * i)
        transfer_vertices_b, _, _ = vae.decoding(z_id_b, z_exp_b - (z_exp_b - z_exp_a) / 150 * i)
        transfer_vertices_a += mean_face
        transfer_vertices_b += mean_face
        vertices_a_transfer = transfer_vertices_a.squeeze(
            0).detach().cpu().numpy() * de_normalise_factor
        vertices_b_transfer = transfer_vertices_b.squeeze(
            0).detach().cpu().numpy() * de_normalise_factor
        vertices_list.append(vertices_a_transfer)
        vertices_list.append(vertices_b_transfer)

    visualise(vis_face=vis_face, vertices_list=vertices_list, show_type="exp_transfer",
              datasets_type=datasets_type)


def expression_interpolate(datasets_type="COMA", ip_num=4, with_gt=True):
    vis_face, two_persons_vertices, de_normalise_factor, mean_face, vae = load_model(
        show_type="exp_interpolate", datasets_type=datasets_type, with_gt=with_gt)
    true_vertices_a_neu = list(two_persons_vertices.values())[0].unsqueeze(0)
    true_vertices_a_exp = list(two_persons_vertices.values())[1].unsqueeze(0)
    true_vertices_b_neu = list(two_persons_vertices.values())[2].unsqueeze(0)
    true_vertices_b_exp = list(two_persons_vertices.values())[3].unsqueeze(0)
    interpolated_vertices_list = []
    vertices_a_neu_gt = true_vertices_a_neu[0].detach().cpu().numpy() * de_normalise_factor
    vertices_b_exp_gt = true_vertices_b_exp[0].detach().cpu().numpy() * de_normalise_factor
    vertices_b_neu_gt = true_vertices_b_neu[0].detach().cpu().numpy() * de_normalise_factor
    interpolated_vertices_list.append(vertices_b_neu_gt)

    z_id_b, z_exp_b_neu = vae.encoding_eval(true_vertices_b_neu)
    _, z_exp_b_exp = vae.encoding_eval(true_vertices_b_exp)
    z_id_a, _ = vae.encoding_eval(true_vertices_a_neu)
    z_exp_diff = z_exp_b_exp - z_exp_b_neu
    z_exp_diff_unit = z_exp_diff / ip_num
    z_id_diff = z_id_a - z_id_b
    z_id_diff_unit = z_id_diff / ip_num

    """   EXP   """
    # i-1 interpolations between the start and end
    # for j in range(1, ip_num):
    #     z_exp_in = z_exp_b_neu + z_exp_diff_unit * j
    #     interpolate_vertices, _, _ = vae.decoding(z_id_b, z_exp_in)
    #     interpolate_vertices += mean_face
    #     inpo_vertices = interpolate_vertices.detach().cpu().numpy() * de_normalise_factor
    #     interpolated_vertices_list.append(np.squeeze(inpo_vertices, axis=0))
    # interpolated_vertices_list.append(vertices_b_exp_gt)

    """   ID   """
    for j in range(1, ip_num):
        z_id_in = z_id_b + z_id_diff_unit * j
        interpolate_vertices, _, _ = vae.decoding(z_id_in, z_exp_b_neu)
        interpolate_vertices += mean_face
        inpo_vertices = interpolate_vertices.detach().cpu().numpy() * de_normalise_factor
        interpolated_vertices_list.append(np.squeeze(inpo_vertices, axis=0))
    interpolated_vertices_list.append(vertices_a_neu_gt)

    visualise(vis_face=vis_face, vertices_list=interpolated_vertices_list, datasets_type=datasets_type, show_type="exp_interpolate")


if __name__ == '__main__':
    if "COMA" in NAME:
        dataset_type = "COMA"
    elif "FaceScape" in NAME:
        dataset_type = "FaceScape"
    else:
        raise ValueError("Please include dataset name in log name, [COMA|FaceScape].")
    expression_transfer(datasets_type=dataset_type)
    expression_interpolate(datasets_type=dataset_type)
