import distutils
import logging
import shutil
from time import time, sleep

import torch
import torch.nn.functional as F
from psbody.mesh import MeshViewers, Mesh
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from Model.vae import VariationalAE
from Model.discriminator import Discriminator
from Model.laplacian_loss import LaplacianLoss
from path import *
from utils.dataset import BU3DFE
from utils.coma_dataset import COMA
from utils.facescape_dataset import FaceScape
from utils.logger import TrainingLogger

KL_WARM_UP = 350


def visualise_faces(epoch, vae_output, mesh_viewer, visualisation, write_obj):
    faces, true_vertices, true_neutral_vertices, true_exp_vertices, mean_face, pred_vertices, pred_vertices_neutral, pred_vertices_exp, _, _ = vae_output

    mesh_vis_true = Mesh(f=faces)
    mesh_vis_true_neutral = Mesh(f=faces)
    mesh_vis_true_exp = Mesh(f=faces)
    mesh_vis_neutral = Mesh(f=faces)
    mesh_vis_exp = Mesh(f=faces)
    mesh_vis_full = Mesh(f=faces)
    mesh_mean_exp = Mesh(f=faces)

    mesh_vis_true.v = true_vertices[0].detach().cpu().numpy()
    mesh_vis_true_neutral.v = true_neutral_vertices[0].detach().cpu().numpy()
    mesh_vis_true_exp.v = (
            true_vertices[0] - true_neutral_vertices[0] + mean_face[0]).detach().cpu().numpy()
    mesh_vis_neutral.v = (pred_vertices_neutral[0] + mean_face[0]).detach().cpu().numpy()
    mesh_vis_exp.v = (pred_vertices_exp[0] + mean_face[0]).detach().cpu().numpy()
    mesh_vis_full.v = (pred_vertices[0] + mean_face[0]).detach().cpu().numpy()
    # if not args.with_gt:
    #     mesh_mean_exp.v = true_exp_vertices[0].detach().cpu().numpy()
    # else:
    #     mesh_mean_exp.v = mesh_vis_true_exp.v
    if visualisation:
        # Visualisation.
        mesh_viewer[1][0].set_dynamic_meshes([mesh_vis_true])
        mesh_viewer[1][1].set_dynamic_meshes([mesh_vis_true_neutral])
        mesh_viewer[1][2].set_dynamic_meshes([mesh_vis_true_exp])
        mesh_viewer[0][0].set_dynamic_meshes([mesh_vis_full])
        mesh_viewer[0][1].set_dynamic_meshes([mesh_vis_neutral])
        mesh_viewer[0][2].set_dynamic_meshes([mesh_vis_exp])
        if not args.with_neutral_gt:
            mesh_mean_exp.v = true_exp_vertices[0].detach().cpu().numpy()
            mesh_viewer[1][3].set_dynamic_meshes([mesh_mean_exp])

    if write_obj:
        if epoch == 1:
            mesh_vis_true.write_obj(
                filename=log_name_dir + "save_obj/sub_vt_gt.obj")
            mesh_vis_true_neutral.write_obj(
                filename=log_name_dir + "save_obj/_id_gt.obj")
            mesh_vis_true_exp.write_obj(
                filename=log_name_dir + "save_obj/sub{0}_exp_gt.obj")
        mesh_vis_full.write_obj(
            filename=log_name_dir + "save_obj/sub_epoch{0}_vt_pred.obj".format(epoch))
        mesh_vis_neutral.write_obj(
            filename=log_name_dir + "save_obj/sub_epoch{0}_id_pred.obj".format(epoch))
        mesh_vis_exp.write_obj(
            filename=log_name_dir + "save_obj/sub_epoch{0}_exp_pred.obj".format(epoch))


def calculate_vae_losses(partition, args, epoch, vae_output, laplacian_loss, id_discriminator, mesh_viewer, visualisation=False, write_obj=False):

    _, true_vertices, true_neutral_vertices, true_exp_vertices, mean_face, pred_vertices, pred_vertices_neutral, pred_vertices_exp, kl_loss_id, kl_loss_exp = vae_output
    vae_batch_size = true_vertices.shape[0]

    visualise_faces(epoch, vae_output, mesh_viewer, visualisation, write_obj)

    # L2 loss.
    if args.with_neutral_gt:
        # Compute full vertices l2 loss and neutral vertices l2 loss
        l2_loss = F.mse_loss(pred_vertices, true_vertices - mean_face) + \
                   args.lambda6 * F.mse_loss(pred_vertices_neutral, true_neutral_vertices - mean_face)
    else:
        # Compute full vertices l2 loss and average expression vertices l2 loss
        l2_loss = F.mse_loss(pred_vertices, true_vertices - mean_face)

    if args.dataset == "BU3DFE" and args.with_neutral_gt:
        l2_loss += args.lambda6 * F.mse_loss(pred_vertices_exp, true_vertices - true_neutral_vertices)

    # Final loss.
    if args.dataset == "COMA" and partition == "train":
        # epochs should be less than KL_WARM_UP, and when the epoch increases, the KL is less important
        kl_loss_id = (KL_WARM_UP - epoch) / KL_WARM_UP * kl_loss_id
        kl_loss_exp = (KL_WARM_UP - epoch) / KL_WARM_UP * kl_loss_exp

    vae_loss = args.lambda1 * l2_loss + args.lambda2 * kl_loss_id + args.lambda3 * kl_loss_exp

    if (not args.with_neutral_gt) or args.dataset == "BU3DFE":
        # Laplacian Loss
        lapla_loss = laplacian_loss(pred_vertices_neutral + mean_face)
        vae_loss += lapla_loss * args.lambda8

    # For saving model in evaluation part
    save_criteria = args.lambda1 * l2_loss.detach().clone()

    # ID discriminator loss.
    if args.id_discriminator_used:
        concatenate_vertices = torch.cat([true_vertices, pred_vertices_neutral + mean_face],
                                         dim=2)
        id_discriminator_prediction_vae = id_discriminator(concatenate_vertices)
        id_discriminator_loss_vae = F.cross_entropy(id_discriminator_prediction_vae,
                                                    torch.ones(
                                                        size=(pred_vertices.shape[0],),
                                                        dtype=torch.long, device=device))
        vae_loss += args.lambda4 * id_discriminator_loss_vae
        # Get discriminator prediction results.
        id_pred_choice_vae = torch.exp(id_discriminator_prediction_vae).max(1)[1]
        id_correct_vae = float(id_pred_choice_vae.cpu().sum().item()) / \
                         id_pred_choice_vae.shape[0]

    # Log losses.
    training_logger.log_batch_loss(
        "L2 Loss", l2_loss.item(), partition, vae_batch_size, part_name="VAE-ID_DIS")
    training_logger.log_batch_loss(
        "ID KL Loss", kl_loss_id.item(), partition, vae_batch_size, part_name="VAE-ID_DIS")
    training_logger.log_batch_loss(
        "EXP KL Loss", kl_loss_exp.item(), partition, vae_batch_size, part_name="VAE-ID_DIS")
    if (not args.with_neutral_gt) or args.dataset == "BU3DFE":
        training_logger.log_batch_loss(
            "LAPLACIAN Loss", lapla_loss.item(), partition, vae_batch_size, part_name="VAE-ID_DIS")
    if args.id_discriminator_used:
        training_logger.log_batch_loss(
            "VAE discriminator Loss", id_discriminator_loss_vae.item(), partition,
            vae_batch_size,
            part_name="VAE-ID_DIS")
        training_logger.log_batch_loss(
            "VAE discriminator Accuracy", id_correct_vae, partition, vae_batch_size,
            part_name="VAE-ID_DIS")

    return vae_loss, save_criteria


def neutral_latent_loss(neutral_latent_vector, neutral_exp_latent_vector, id_latent_vector):
    neu_latent_loss = F.l1_loss(neutral_latent_vector, id_latent_vector)
    return neu_latent_loss


def train(args, training_logger):
    if args.dataset == "COMA":
        coma_mean_face_np = np.load(DATASET_PATH + 'coma_average_vertices_fixed.npy')
        coma_mean_face = torch.from_numpy(coma_mean_face_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        mean_face = coma_mean_face
        train_data = COMA(partition='train', always_sample_same_id=True)
        test_data = COMA(partition='test')
    elif args.dataset == "BU3DFE":
        bu3dfe_mean_face_np = np.load(DATASET_PATH + 'BU3DFE_mean_face_10f.npy')
        bu3dfe_mean_face = torch.from_numpy(bu3dfe_mean_face_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        mean_face = bu3dfe_mean_face
        train_data = BU3DFE(partition='train', always_sample_same_id=True)
        test_data = BU3DFE(partition='test')
    elif args.dataset == "FaceScape":
        facescape_mean_face_np = np.load(DATASET_PATH + 'facescape_train_mean_face_70percent.npy')
        facescape_mean_face = torch.from_numpy(facescape_mean_face_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        mean_face = facescape_mean_face
        train_data = FaceScape(partition='train', always_sample_same_id=True)
        test_data = FaceScape(partition='test')
    else:
        raise ValueError("No such dataset.")

    train_loader = DataLoader(train_data, num_workers=0, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=0, batch_size=args.test_batch_size, shuffle=True)

    vae = VariationalAE(args, train_data.vertices_num, args.latent_vector_dim_id, args.latent_vector_dim_exp).to(device)
    optimiser_vae = torch.optim.Adam(vae.parameters(), lr=args.lr_vae, weight_decay=1e-4)
    id_discriminator = Discriminator(args, train_data.vertices_num, p=args.p_dropout, global_feat=True).to(device)
    optimiser_id_discriminator = torch.optim.Adam(id_discriminator.parameters(), lr=args.lr_id_discriminator)

    scheduler_vae = StepLR(optimizer=optimiser_vae, step_size=50, gamma=0.7)
    scheduler_id_discriminator = StepLR(optimizer=optimiser_id_discriminator, step_size=50, gamma=0.7)

    if args.dataset == "COMA":
        id_discriminator.load_state_dict(torch.load(LOGS_PATH + "ID_discriminator_COMA/_model.pt"), strict=False)
    elif args.dataset == "BU3DFE":
        id_discriminator.load_state_dict(torch.load(LOGS_PATH + "ID_discriminator_BU3DFE_10f/_model.pt"), strict=False)
    elif args.dataset == "FaceScape":
        id_discriminator.load_state_dict(torch.load(LOGS_PATH + "ID_discriminator_FaceScape_70percent/_model.pt"), strict=False)
    else:
        raise ValueError("No such dataset.")

    logging.info("Start training...")
    full_start_time = time()

    if args.dataset == "COMA":
        mesh_viewer = MeshViewers(shape=(2, 4))
        coma_faces = np.load(DATASET_PATH + "coma_faces.npy")
        faces = coma_faces
    elif args.dataset == "BU3DFE":
        mesh_viewer = MeshViewers(shape=(2, 4))
        bu3dfe_faces = np.load(DATASET_PATH + "BU3DFE_face.npy")
        faces = bu3dfe_faces
    elif args.dataset == "FaceScape":
        mesh_viewer = MeshViewers(shape=(2, 4))
        facescape_faces = np.load(DATASET_PATH + "facescape_faces.npy")
        faces = facescape_faces
    else:
        raise ValueError("No such dataset.")

    laplacian_loss = LaplacianLoss(vertices_number=train_data.vertices_num, faces=faces, average=True)

    def model_save_specific_epochs(log_dict, part_name=""):
        specific_epoch = log_dict["epoch"]
        torch.save(log_dict["model_weights"], log_name_dir + part_name + "_model_{}.pt".format(specific_epoch))
        logging.info("{} epoch model weights for ".format(specific_epoch) + part_name + " are saved.")

    for epoch in range(1, args.epochs + 1):
        logging.info("*** Epoch " + str(epoch) + " ***")
        epoch_start_time = time()

        """ VAE """
        """ Train """
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            if args.dataset == "BU3DFE":
                true_vertices, _, _, _, _, _, sample_id_vertices, id_label_same, _, _, true_neutral_vertices, true_exp_vertices = data
            elif args.dataset == "COMA":
                true_vertices, _, expression_levels, _, true_neutral_vertices, true_exp_vertices, sample_id_vertices, id_label_same, _, _ = data
                # Neutral
                # expressions[expression_levels <= 0.05] = 16
            elif args.dataset == "FaceScape":
                true_vertices, _, _, sample_id_vertices, id_label_same, _, _, true_neutral_vertices, true_exp_vertices = data
            else:
                raise ValueError("No such dataset.")

            true_concatenate_vertices = torch.cat([true_vertices, sample_id_vertices], dim=2)
            if args.dataset == "COMA":
                try:
                    pred_vertices, pred_vertices_neutral, pred_vertices_exp, kl_loss_id, kl_loss_exp, z_id, _ = vae(true_vertices, expression_levels<=0.1)
                except ValueError:
                    pass
            else:
                pred_vertices, pred_vertices_neutral, pred_vertices_exp, kl_loss_id, kl_loss_exp, z_id, _ = vae(
                    true_vertices)
            pred_concatenate_vertices = torch.cat([true_vertices, pred_vertices_neutral + mean_face], dim=2)

            vae_output = [faces, true_vertices, true_neutral_vertices, true_exp_vertices, mean_face,
                          pred_vertices, pred_vertices_neutral, pred_vertices_exp, kl_loss_id,
                          kl_loss_exp]

            """ Discriminator """
            if args.id_discriminator_used:
                """ ID Discriminator """
                id_discriminator.train()

                # Build combined batch.
                assert torch.all(id_label_same)
                combined_concatenate_vertices = torch.cat(
                    [true_concatenate_vertices, pred_concatenate_vertices], dim=0)
                combined_label = torch.cat([id_label_same, torch.zeros_like(id_label_same)])

                # Shuffle this batch.
                shuffle_idx = torch.randperm(combined_label.shape[0])
                combined_concatenate_vertices = combined_concatenate_vertices[shuffle_idx]
                combined_label = combined_label[shuffle_idx]

                # Forward pass.
                optimiser_id_discriminator.zero_grad()
                id_pred_log_softmax = id_discriminator(combined_concatenate_vertices)
                id_loss = F.cross_entropy(id_pred_log_softmax, combined_label)

                # Backward pass.
                id_loss.backward(retain_graph=True)
                optimiser_id_discriminator.step()
                id_pred_choice = torch.exp(id_pred_log_softmax).max(1)[1]
                id_correct = id_pred_choice.eq(combined_label.data).cpu().sum()

                # Log losses.
                training_logger.log_batch_loss(
                    "ID Dis loss", id_loss.item(), "train", id_pred_log_softmax.shape[0],
                    part_name="VAE-ID_DIS")
                training_logger.log_batch_loss(
                    "ID_accuracy", id_correct.item() / id_pred_log_softmax.shape[0],
                    "train", id_pred_log_softmax.shape[0], part_name="VAE-ID_DIS")
                training_logger.log_batch_loss(
                    "ID accuracy true", -1,
                    "train", id_pred_log_softmax.shape[0], part_name="VAE-ID_DIS")
                training_logger.log_batch_loss(
                    "ID accuracy vae", -1,
                    "train", id_pred_log_softmax.shape[0], part_name="VAE-ID_DIS")

                if args.save_id_dis_model:
                    if epoch % args.epochs == 0 and i == len(train_loader)-1:
                        model_save_specific_epochs({"epoch": epoch,
                                                    "model_weights": id_discriminator.state_dict()},
                                                   part_name="DISCRIMINATOR")

            """ Generator """
            vae.train()
            # Forward pass.
            optimiser_vae.zero_grad()
            vae_batch_size = true_vertices.shape[0]

            loss, _ = calculate_vae_losses(partition="train", args=args, epoch=epoch,
                                            vae_output=vae_output, laplacian_loss=laplacian_loss,
                                            id_discriminator=id_discriminator, visualisation=args.visualise,
                                            mesh_viewer=mesh_viewer)
            if args.dataset == "COMA":
                _, _, _, _, _, z_neutral, z_neutral_exp = vae(pred_vertices_neutral + mean_face,
                                                              expression_levels<=0.1)
            else:
                _, _, _, _, _, z_neutral, z_neutral_exp = vae(pred_vertices_neutral + mean_face)
            id_latent_loss = neutral_latent_loss(neutral_latent_vector=z_neutral,
                                                 neutral_exp_latent_vector=z_neutral_exp,
                                                 id_latent_vector=z_id)
            loss += id_latent_loss * args.lambda7

            training_logger.log_batch_loss("NEUTRAL LATENT L1 Loss", id_latent_loss.item(),
                                           partition="train", size=vae_batch_size, part_name="VAE-ID_DIS")

            training_logger.log_batch_loss("loss", loss.item(), partition="train",
                                           size=vae_batch_size, part_name="VAE-ID_DIS")

            # Backward pass.
            loss.backward()
            optimiser_vae.step()

        if args.id_discriminator_used:
            scheduler_id_discriminator.step()
        scheduler_vae.step()

        """ Eval """
        with torch.no_grad():
            for data_idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                if args.dataset == "BU3DFE":
                    true_vertices_eval, _, _, _, _, _, sample_id_vertices_eval, id_label_same_eval, _, _, true_neutral_vertices_eval, true_exp_vertices_eval = data
                elif args.dataset == "COMA":
                    true_vertices_eval, _, expression_levels_eval, _, true_neutral_vertices_eval, true_exp_vertices_eval, sample_id_vertices_eval, id_label_same_eval, _, _ = data
                    # expressions_eval[expression_levels_eval <= 0.05] = 16
                elif args.dataset == "FaceScape":
                    true_vertices_eval, _, _, sample_id_vertices_eval, id_label_same_eval, _, _, true_neutral_vertices_eval, true_exp_vertices_eval = data
                else:
                    raise ValueError("No such dataset.")

                vae_batch_size = true_vertices_eval.shape[0]

                pred_vertices_eval, pred_vertices_neutral_eval, pred_vertices_exp_eval, kl_loss_id_eval, kl_loss_exp_eval, z_id_eval, _ = vae(true_vertices_eval)
                true_concatenate_vertices_eval = torch.cat(
                    [true_vertices_eval, sample_id_vertices_eval], dim=2)
                pred_concatenate_vertices_eval = torch.cat([true_vertices_eval.to(device),
                                                       pred_vertices_neutral_eval + mean_face],
                                                      dim=2)
                vae_output = [faces, true_vertices_eval, true_neutral_vertices_eval, true_exp_vertices_eval,
                              mean_face, pred_vertices_eval, pred_vertices_neutral_eval, pred_vertices_exp_eval,
                              kl_loss_id_eval, kl_loss_exp_eval]

                """ Discriminator """
                if args.id_discriminator_used:
                    id_discriminator.eval()

                    # Forward pass.
                    id_pred_log_softmax_true = id_discriminator(true_concatenate_vertices_eval)
                    id_loss_true = F.cross_entropy(id_pred_log_softmax_true, id_label_same_eval)
                    id_pred_choice_true = torch.exp(id_pred_log_softmax_true).max(1)[1]
                    id_correct_true = id_pred_choice_true.eq(id_label_same_eval.data).cpu().sum()
                    id_accuracy_true = id_correct_true.item() / \
                                       id_pred_log_softmax_true.shape[0]

                    id_pred_log_softmax_predict = id_discriminator(
                        pred_concatenate_vertices_eval)
                    id_loss_predict = F.cross_entropy(id_pred_log_softmax_predict,
                                                      torch.zeros_like(id_label_same_eval))
                    id_pred_choice_predict = torch.exp(id_pred_log_softmax_predict).max(1)[1]
                    id_correct_predict = id_pred_choice_predict.eq(
                        torch.zeros_like(id_label_same_eval).data).cpu().sum()
                    id_accuracy_predict = id_correct_predict.item() / \
                                          id_pred_log_softmax_predict.shape[0]

                    # Log eval losses.
                    training_logger.log_batch_loss(
                        "ID Dis loss", (id_loss_true.item() + id_loss_predict) / 2, "eval",
                        id_pred_log_softmax_true.shape[0], part_name="VAE-ID_DIS")
                    training_logger.log_batch_loss(
                        "ID_accuracy", (id_accuracy_true + id_accuracy_predict) / 2,
                        "eval", id_pred_log_softmax_true.shape[0],
                        part_name="VAE-ID_DIS")
                    training_logger.log_batch_loss(
                        "ID accuracy true", id_accuracy_true,
                        "eval", id_pred_log_softmax_true.shape[0],
                        part_name="VAE-ID_DIS")
                    training_logger.log_batch_loss(
                        "ID accuracy vae", id_accuracy_predict,
                        "eval", id_pred_log_softmax_true.shape[0],
                        part_name="VAE-ID_DIS")

                """ Generator """
                vae.eval()
                Written_Flag = False
                if data_idx == 9 and Written_Flag:
                    _, param_l2_loss = calculate_vae_losses(partition="eval", args=args,
                                                             epoch=epoch, vae_output=vae_output,
                                                             laplacian_loss=laplacian_loss,
                                                             id_discriminator=id_discriminator,
                                                             mesh_viewer=mesh_viewer, write_obj=True)
                else:
                    _, param_l2_loss = calculate_vae_losses(partition="eval", args=args,
                                                            epoch=epoch,
                                                            vae_output=vae_output,
                                                            laplacian_loss=laplacian_loss,
                                                            id_discriminator=id_discriminator,
                                                            mesh_viewer=mesh_viewer, 
                                                            visualisation=args.visualise)

                _, _, _, _, _, z_neutral_eval, z_neutral_exp_eval = vae(pred_vertices_neutral_eval + mean_face)
                id_latent_loss = neutral_latent_loss(neutral_latent_vector=z_neutral_eval,
                                                     neutral_exp_latent_vector=z_neutral_exp_eval,
                                                     id_latent_vector=z_id_eval)

                # No consider the id discriminator loss when saving models
                param_l2_loss += id_latent_loss * args.lambda7

                training_logger.log_batch_loss("NEUTRAL LATENT L1 Loss", id_latent_loss.item(),
                                               partition="eval", size=vae_batch_size,
                                               part_name="VAE-ID_DIS")
                training_logger.log_batch_loss("loss", param_l2_loss.item(), partition="eval",
                                               size=vae_batch_size, part_name="VAE-ID_DIS")

            training_logger.log_epoch({"epoch": epoch,
                                       "model_weights": vae.state_dict(),
                                       "optimiser_weights": optimiser_vae.state_dict()}, part_name="VAE-ID_DIS")
            if epoch % args.epochs == 0:
                model_save_specific_epochs({"epoch": epoch,
                                       "model_weights": vae.state_dict()}, part_name="VAE-ID_DIS")


        epoch_end_time = time()
        logging.info("Time left: " + time_format((epoch_end_time - epoch_start_time) * (args.epochs - epoch)))
        torch.cuda.empty_cache()

    full_end_time = time()
    logging.info("Total training time: " + time_format(full_end_time - full_start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Disentanglement.")

    parser.add_argument("-n", "--name", type=str, default="Temporary Experiment",
                        help="Name of the experiment.")
    parser.add_argument("--latent_vector_dim_id", type=int, default=160, metavar="N",
                        help="ID latent vector dimensions.")
    parser.add_argument("--latent_vector_dim_exp", type=int, default=160, metavar="N",
                        help="EXP latent vector dimensions.")
    parser.add_argument("-d", "--dataset", type=str, default="BU3DFE",
                        help="Datasets: BU3DFE or COMA or FaceScape.")
    parser.add_argument("--lr_vae", type=float, default=1e-4,
                        help="VAE learning rate.")
    parser.add_argument("--lr_id_discriminator", type=float, default=1e-4,
                        help="ID discriminator learning rate.")
    parser.add_argument("--id_discriminator_used", type=lambda x:bool(distutils.util.strtobool(x)), default=True,
                        help="Using ID discriminator.")
    parser.add_argument("--input_channel", type=int, default=6, metavar="N",
                        help="Input channel.")
    parser.add_argument("-e", "--epochs", type=int, default=50, metavar="N",
                        help="Number of epochs to train.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, metavar="N",
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", type=int, default=16, metavar="N",
                        help="Batch size for evaluating.")
    parser.add_argument("--lambda1", type=float, default=1e2,
                        help="Lambda to balance l2 loss function.")
    parser.add_argument("--lambda2", type=float, default=5e-7,
                        help="Lambda to balance ID KL loss functions.")
    parser.add_argument("--lambda3", type=float, default=5e-7,
                        help="Lambda to balance EXP KL loss functions.")
    parser.add_argument("--lambda4", type=float, default=1e-3,
                        help="Lambda to balance ID discriminator loss functions.")
    parser.add_argument("--lambda6", type=float, default=10,
                        help="Lambda to balance Neutral face L2 loss.")
    parser.add_argument("--lambda7", type=float, default=1e-2,
                        help="Lambda of ID_LATENT_LAMBDA.")
    parser.add_argument("--lambda8", type=float, default=1e-2,
                        help="Lambda of Laplacian loss.")
    parser.add_argument("-p", "--p_dropout", type=float, default=0.5, metavar="N",
                        help="P for dropout.")
    parser.add_argument("--with_neutral_gt", type=lambda x:bool(distutils.util.strtobool(x)), default=False,
                        help="Using neutral faces ground truth.")
    parser.add_argument("--use_bn", type=lambda x:bool(distutils.util.strtobool(x)), default=False,
                        help="Using bn and dropout in the ID discriminator network.")
    parser.add_argument("--save_id_dis_model", type=lambda x:bool(distutils.util.strtobool(x)), default=False,
                        help="Saving ID discriminator Model.")
    parser.add_argument("--visualise", type=lambda x:bool(distutils.util.strtobool(x)), default=True,
                        help="Visualisation.")

    args = parser.parse_args()

    """ args override zone starts. """
    # args.name = "BU3DFE_wogt_wobndropout_withsoftmax"
    # args.dataset = "BU3DFE"
    # args.with_neutral_gt = False

    # if args.dataset == "BU3DFE":
    #     if args.id_discriminator_used:
    #         args.name = "BU3DFE_using_iddis_using_gt"
    #     else:
    #         args.name = "BU3DFE_wo_iddis_using_gt"
    #     args.latent_vector_dim_id = 40
    #     args.latent_vector_dim_exp = 40
    #     args.lambda1 = 250
    #     args.lambda2 = 3e-5
    #     args.lambda3 = 3e-5
    #     args.lambda6 = 0.5
    #     args.lambda7 = 5e-2
    #     args.lambda8 = 10
    #     args.use_bn = False
    #     args.batch_size = 8
    #     args.epochs = 280
    #
    # elif args.dataset == "COMA":
    #     args.latent_vector_dim_id = 4
    #     args.latent_vector_dim_exp = 4
    #     if args.with_neutral_gt:
    #         args.lr_id_discriminator = 1e-3
    #         args.lambda1 = 5000
    #         args.lambda2 = 3e-3
    #         args.lambda3 = 3e-3
    #         args.lambda4 = 1e-5
    #         args.lambda6 = 5
    #         args.lambda7 = 7e-3
    #         args.lambda8 = 7e-3
    #     else:
    #         args.lambda1 = 600
    #         args.lambda4 = 5e-4
    #         args.lambda7 = 5e-3
    #         args.use_bn = False
    #     args.batch_size = 32
    #     args.epochs = 300
    #
    # elif args.dataset == "FaceScape":
    #     args.latent_vector_dim_id = 64
    #     args.latent_vector_dim_exp = 64
    #
    #     args.lambda1 = 5000
    #     args.lambda2 = 3e-5
    #     args.lambda3 = 3e-5
    #     args.lambda4 = 5e-4
    #     args.lambda6 = 1
    #
    #     args.use_bn = False
    #     args.batch_size = 32
    #     args.epochs = 280
    """ args override zone ends. """

    # Setup logging.
    # Initialise log folder.
    log_name_dir = LOGS_PATH + args.name + "/"
    if os.path.exists(log_name_dir):
        if args.name == "Temporary Experiment":
            shutil.rmtree(log_name_dir)
            shutil.rmtree(LOGS_PATH + "tf_board/Temporary Experiment")
        else:
            raise ValueError("Name has been used.")
    os.mkdir(log_name_dir)

    # Initialise logging config.
    logging_level = logging.INFO

    console_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=log_name_dir + "log.txt"), console_handler],
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
    console_handler.setLevel(logging_level)

    logging.info(vars(args))

    training_logger = TrainingLogger(log_name_dir, args)

    train(args, training_logger)