import json
import logging
import shutil
import time
from types import SimpleNamespace
import sys

from matplotlib import pyplot as plt
from psbody.mesh import MeshViewers, Mesh
from sklearn import svm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

from Model.vae import VariationalAE
from utils.coma_dataset import COMA
from utils.dataset import BU3DFE
from utils.facescape_dataset import FaceScape
from path import *

# NAME = "COMA_lc4_epoch300_gt_l65_woiddis_woinfobn_noneexplc_twice"
NAME = sys.argv[1]


def evaluate(write_obj=False, datasets_type="COMA"):
    logging.info(NAME)

    # Create save folder.
    eval_write_path = LOGS_PATH + NAME + "/eval/"
    if os.path.exists(eval_write_path):
        shutil.rmtree(eval_write_path)
    os.mkdir(eval_write_path)

    # Load data.
    if datasets_type == "COMA":
        test_data = COMA(partition='test')
        test_loader = DataLoader(test_data, num_workers=0, batch_size=1, shuffle=False)
        de_normalise_factor = COMA_NORMALISE * 1000
        mean_face = torch.from_numpy(
            np.load(DATASET_PATH + 'coma_average_vertices_fixed.npy')).to(device=device, dtype=torch.float32).unsqueeze(0)
        coma_faces = np.load(DATASET_PATH + "coma_faces.npy")
        vis_face = coma_faces
    elif datasets_type == "BU3DFE":
        test_data = BU3DFE(partition="test", sort=True)
        test_loader = DataLoader(test_data, num_workers=0, batch_size=1, shuffle=False)
        de_normalise_factor = BU3DFE_NORMALISE
        mean_face = torch.from_numpy(np.load(DATASET_PATH + 'BU3DFE_mean_face.npy')).to(device=device, dtype=torch.float32).unsqueeze(0)
        bu3dfe_faces = np.load(DATASET_PATH + "BU3DFE_face.npy")
        vis_face = bu3dfe_faces
    elif datasets_type == "FaceScape":
        test_data = FaceScape(partition='test')
        test_loader = DataLoader(test_data, num_workers=0, batch_size=1, shuffle=False)
        de_normalise_factor = FACESCAPE_NORMALISE
        mean_face = torch.from_numpy(np.load(DATASET_PATH + 'facescape_train_mean_face.npy')).to(device=device, dtype=torch.float32).unsqueeze(0)
        facescape_faces = np.load(DATASET_PATH + "facescape_faces.npy")
        vis_face = facescape_faces
    else:
        raise ValueError("No such dataset.")

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


    # Load prerequisite information.
    with open(LOGS_PATH + NAME + "/config.json", "r") as f:
        args = SimpleNamespace(**json.load(f))

    # Setup for mesh visualisation.
    mesh_viewer = MeshViewers(shape=(2, 3))
    mesh_vis_true = Mesh(f=vis_face)
    mesh_vis_true_neutral = Mesh(f=vis_face)
    mesh_vis_true_exp = Mesh(f=vis_face)
    mesh_vis_neutral = Mesh(f=vis_face)
    mesh_vis_exp = Mesh(f=vis_face)
    mesh_vis_full = Mesh(f=vis_face)

    # Load trained model.
    vae = VariationalAE(args, test_data.vertices_num, args.latent_vector_dim_id, args.latent_vector_dim_exp).to(
            device)
    vae.load_state_dict(torch.load(LOGS_PATH + NAME + "/VAE-ID_DIS_model_{0}.pt".format(args.epochs)))
    vae.eval()

    logging.info("Total number of parameters: " + str(sum(p.numel() for p in vae.parameters() if p.requires_grad)))

    z_ids = []
    z_exps = []
    all_subject_ids = []
    avg_distances = []
    ne_avg_distances = []
    same_id_ne_v = {}
    for data in tqdm(test_loader):
        # Obtain data.
        if datasets_type == "COMA":
            true_vertices, expressions, expression_levels, subject_ids, true_neutral_vertices, true_exp_vertices, _, _, _, _ = data
            subject_ids = subject_ids.item()
        elif datasets_type == "BU3DFE":
            true_vertices, expressions, expression_levels, gender, _, subject_ids, _, _, _, _, true_neutral_vertices, true_exp_vertices = data
            subject_ids = subject_ids.item()
            if gender.item() == 1:
                subject_ids = subject_ids + 100
        elif datasets_type == "FaceScape":
            true_vertices, expressions, subject_ids, _, _, _, _, true_neutral_vertices, true_exp_vertices = data
            subject_ids = subject_ids.item()
        else:
            raise ValueError("No such dataset.")


        # Obtain prediction results.
        z_id, z_exp = vae.encoding_eval(true_vertices)
        pred_vertices, pred_vertices_neutral, pred_vertices_exp = vae.decoding(z_id, z_exp)

        pred_vertices += mean_face
        pred_vertices_neutral += mean_face
        pred_vertices_exp += mean_face

        # Prepare data on CPU.
        z_id = z_id[0].detach().cpu().numpy()
        z_exp = z_exp[0].detach().cpu().numpy()
        vertices_gt = true_vertices[0].detach().cpu().numpy() * de_normalise_factor
        vertices_ne_gt = true_neutral_vertices[0].detach().cpu().numpy() * de_normalise_factor
        vertices_exp_gt = \
            (true_vertices - true_neutral_vertices + mean_face)[0].detach().cpu().numpy() * de_normalise_factor
        vertices_pred = pred_vertices[0].detach().cpu().numpy() * de_normalise_factor
        vertices_ne_pred = pred_vertices_neutral[0].detach().cpu().numpy() * de_normalise_factor
        vertices_exp_pred = pred_vertices_exp[0].detach().cpu().numpy() * de_normalise_factor

        # Rigid registration.
        if args.dataset == "BU3DFE":
            R, T = compute_rt(vertices_pred, vertices_gt)
            vertices_pred = vertices_pred @ R.T + T
            R, T = compute_rt(vertices_ne_pred, vertices_ne_gt)
            vertices_ne_pred = vertices_ne_pred @ R.T + T

        # Calculate Euclidean distances.
        avg_distance = np.sqrt(((vertices_gt - vertices_pred)
                                ** 2).sum(1)).mean()
        avg_distances.append(avg_distance)
        ne_avg_distance = np.sqrt(((vertices_ne_gt - vertices_ne_pred)
                                   ** 2).sum(1)).mean()

        ne_avg_distances.append(ne_avg_distance)


        if subject_ids in same_id_ne_v:
            same_id_ne_v[subject_ids].append(vertices_ne_pred)
        else:
            same_id_ne_v[subject_ids] = [vertices_ne_pred]

        z_ids.append(z_id)
        z_exps.append(z_exp)
        all_subject_ids.append(subject_ids)

        # Visualisation.
        mesh_vis_true.v = vertices_gt
        mesh_vis_true_neutral.v = vertices_ne_gt
        mesh_vis_true_exp.v = vertices_exp_gt
        mesh_vis_full.v = vertices_pred
        mesh_vis_neutral.v = vertices_ne_pred
        mesh_vis_exp.v = vertices_exp_pred
        mesh_viewer[1][0].set_dynamic_meshes([mesh_vis_true])
        mesh_viewer[1][1].set_dynamic_meshes([mesh_vis_true_neutral])
        mesh_viewer[1][2].set_dynamic_meshes([mesh_vis_true_exp])
        mesh_viewer[0][0].set_dynamic_meshes([mesh_vis_full])
        mesh_viewer[0][1].set_dynamic_meshes([mesh_vis_neutral])
        mesh_viewer[0][2].set_dynamic_meshes([mesh_vis_exp])
        # time.sleep(0.05)

        # Save obj files.
        if write_obj:
            def avg_euc_dis(v_a, v_b):
                return np.sqrt(np.sum((v_a - v_b) ** 2, 1)).mean()

            # Save prediction OBJ results.
            if datasets_type == "FaceScape":
                mesh_vis_true.write_obj(eval_write_path + str(subject_ids) + "_" +
                                        str(expressions.item()) + ".obj")
                mesh_vis_true_exp.write_obj(eval_write_path + str(subject_ids) + "_" +
                                            str(expressions.item()) + "_exp.obj")
                mesh_vis_full.write_obj(eval_write_path + str(subject_ids) + "_" +
                                        str(expressions.item()) + "_pred_" +
                                        "%.3f" % avg_euc_dis(vertices_pred, vertices_gt) + ".obj")
                mesh_vis_neutral.write_obj(eval_write_path + str(subject_ids) + "_" +
                                           str(expressions.item()) + "_ne_pred_" +
                                           "%.3f" % avg_euc_dis(vertices_ne_pred,
                                                                vertices_ne_gt) + ".obj")
                mesh_vis_exp.write_obj(eval_write_path + str(subject_ids) + "_" +
                                       str(expressions.item()) + "_exp_pred_" +
                                       "%.3f" % avg_euc_dis(vertices_exp_pred,
                                                            vertices_exp_gt) + ".obj")
            else:
                if datasets_type == "COMA":
                    mesh_vis_true.v /= 1000
                    mesh_vis_true_neutral.v /= 1000
                    mesh_vis_true_exp.v /= 1000
                    mesh_vis_full.v /= 1000
                    mesh_vis_neutral.v /= 1000
                    mesh_vis_exp.v /= 1000

                mesh_vis_true.write_obj(eval_write_path + str(subject_ids) + "_" +
                                        str(expressions.item()) + "_" +
                                        "%.3f" % expression_levels.item() + ".obj")
                mesh_vis_true_exp.write_obj(eval_write_path + str(subject_ids) + "_" +
                                            str(expressions.item()) + "_" +
                                            "%.3f" % expression_levels.item() + "_exp.obj")
                mesh_vis_full.write_obj(eval_write_path + str(subject_ids) + "_" +
                                        str(expressions.item()) + "_" +
                                        "%.3f" % expression_levels.item() + "_pred_" +
                                        "%.3f" % avg_euc_dis(vertices_pred, vertices_gt) + ".obj")
                mesh_vis_neutral.write_obj(eval_write_path + str(subject_ids) + "_" +
                                           str(expressions.item()) + "_" +
                                           "%.3f" % expression_levels.item() + "_" +
                                           "%.3f" % avg_euc_dis(vertices_ne_pred, vertices_ne_gt) + "_ne_pred.obj")
                mesh_vis_exp.write_obj(eval_write_path + str(subject_ids) + "_" +
                                       str(expressions.item()) + "_" +
                                       "%.3f" % expression_levels.item() + "_" +
                                       "%.3f" % avg_euc_dis(vertices_exp_pred,
                                                            vertices_exp_gt) + "_exp_pred.obj")
                if datasets_type == "COMA":
                    mesh_vis_true.v *= 1000
                    mesh_vis_true_neutral.v *= 1000
                    mesh_vis_true_exp.v *= 1000
                    mesh_vis_full.v *= 1000
                    mesh_vis_neutral.v *= 1000
                    mesh_vis_exp.v *= 1000
            if datasets_type == "COMA":
                mesh_vis_true.v /= 1000
                mesh_vis_true_neutral.v /= 1000
                mesh_vis_true_exp.v /= 1000
                mesh_vis_full.v /= 1000
                mesh_vis_neutral.v /= 1000
                mesh_vis_exp.v /= 1000
            mesh_vis_true_neutral.write_obj(eval_write_path + str(subject_ids) + "_ne.obj")

    # Turn results into numpy arrays.
    avg_distances = np.array(avg_distances)
    ne_avg_distances = np.array(ne_avg_distances)
    z_ids = np.stack(z_ids)
    z_exps = np.stack(z_exps)
    all_subject_ids = np.array(all_subject_ids)

    # Output evaluation results.
    logging.info("Average distance: " + str(np.mean(avg_distances)) +
                 ", std: " + str(np.std(avg_distances)) +
                 ", median: " + str(np.median(avg_distances)))

    logging.info("Average Neutral distance: " + str(np.mean(ne_avg_distances)) +
                 ", std: " + str(np.std(ne_avg_distances)) +
                 ", median: " + str(np.median(ne_avg_distances)))

    def all_euc_std(ne_arr):
        return np.std(np.sqrt(np.sum((ne_arr - ne_arr.mean(0)) ** 2, axis=2)))
    same_id_ne_v = {k: np.stack(v) for k, v in same_id_ne_v.items()}
    all_euc_3dv = [all_euc_std(i) for i in same_id_ne_v.values()]
    logging.info("Within class reconstructed neutral face std 3DV mean: " +
                 str(np.mean(all_euc_3dv)) + ", median: " + str(np.median(all_euc_3dv)))
    between_class_std = []
    for i in range(100):
        between_class_std.append(
            all_euc_std(np.stack([v[np.random.randint(0, v.shape[0])] for v in same_id_ne_v.values()])))
    logging.info("Between class reconstructed neutral face std 3DV mean: " + str(np.mean(between_class_std)) +
                 ", median: " + str(np.median(between_class_std)))

    logging.info("Visualising z_id.")
    cluster_visualise(z_ids, all_subject_ids, sample=-1, save=False)


def cluster_visualise(zs, z_id_labels, sample=10, save=False):
    pca = PCA(n_components=2, whiten=True)
    z_ids_transformed = pca.fit_transform(zs)

    if save:
        with open(LOGS_PATH + NAME + "scatter.csv", "w") as f:
            f.write("x y id\n" + "\n".join(
                [" ".join([str(k) for k in i]) + " " + str(j) for i, j in zip(z_ids_transformed, z_id_labels)]))

    clf = svm.SVC()
    clf.fit(zs, z_id_labels)
    logging.info("SVM classification accuracy: " + str(np.sum(clf.predict(zs) == z_id_labels) / z_id_labels.shape[0]))

    for i in np.unique(z_id_labels):
        if sample > 0:
            points = z_ids_transformed[z_id_labels == i][
                np.random.randint(z_ids_transformed[z_id_labels == i].shape[0], size=sample)]
        else:
            points = z_ids_transformed[z_id_labels == i]
        plt.scatter(points[:, 0], points[:, 1], marker="$" + hex(i).upper()[2:] + "$")
    plt.show()


if __name__ == '__main__':
    if os.path.exists(LOGS_PATH + NAME + "/eval.txt"):
        os.remove(LOGS_PATH + NAME + "/eval.txt")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=LOGS_PATH + NAME + "/eval.txt"), console_handler],
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")

    if "COMA" in NAME:
        dataset_type = "COMA"
    elif "BU3DFE" in NAME:
        dataset_type = "BU3DFE"
    elif "FaceScape" in NAME:
        dataset_type = "FaceScape"
    else:
        raise ValueError("Please include dataset name in log name, [COMA|BU3DFE|FaceScape].")
    evaluate(write_obj=False, datasets_type=dataset_type)
