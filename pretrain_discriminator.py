import logging
import shutil
from time import time
from time import sleep

import torch
import torch.nn.functional as F
from psbody.mesh import MeshViewers, Mesh
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

from Model.discriminator import Discriminator
from path import *
from utils.dataset import BU3DFE
from utils.coma_dataset import COMA
from utils.facescape_dataset import FaceScape
from utils.logger import TrainingLogger


def discriminator_train(args, training_logger):
    # Initialise dataset and dataloader.
    if args.dataset == "BU3DFE":
        train_data = BU3DFE(partition='train')
        test_data = BU3DFE(partition='test')
    elif args.dataset == "COMA":
        train_data = COMA(partition='train')
        test_data = COMA(partition='test')
    elif args.dataset == "FaceScape":
        train_data = FaceScape(partition='train')
        test_data = FaceScape(partition='test')
    else:
        raise ValueError("No such dataset.")

    train_loader = DataLoader(train_data, num_workers=0, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=0, batch_size=args.test_batch_size, shuffle=False)

    # discriminator = Discriminator(args, train_data.vertices_num, p=args.p_dropout).to(device)
    # PointNet++
    discriminator = Discriminator(args, train_data.vertices_num, p=args.p_dropout, global_feat=True).to(device)

    optimiser = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    logging.info("Start training...")
    full_start_time = time()

    mesh_viewer = MeshViewers(shape=(1, 3))

    def model_save_specific_epochs(log_dict, part_name=""):
        specific_epoch = log_dict["epoch"]
        torch.save(log_dict["model_weights"], log_name_dir + part_name + "_model.pt".format(specific_epoch))
        logging.info("{} epoch model weights for ".format(specific_epoch) + part_name + " are saved.")

    for epoch in range(1, args.epochs + 1):
        logging.info("*** Epoch " + str(epoch) + " ***")
        epoch_start_time = time()
        """ Train """
        for data in tqdm(train_loader):
            if args.dataset == "BU3DFE":
                true_vertices, _, _, _, _, _, \
                    sample_id_vertices, id_label_same, sample_exp_vertices, exp_label_same, _, _ = data
            elif args.dataset == "COMA":
                true_vertices, _, _, _, _, _, sample_id_vertices, id_label_same, sample_exp_vertices, exp_label_same = data
            elif args.dataset == "FaceScape":
                true_vertices, _, _, sample_id_vertices, id_label_same, sample_exp_vertices, exp_label_same, _, _ = data
            else:
                raise ValueError("No such dataset.")

            if true_vertices.shape[0] == 1:
                continue

            sample_vertices = sample_id_vertices
            label_same = id_label_same

            if args.dataset == "COMA":
                faces = np.load(DATASET_PATH + "coma_faces.npy")
                mesh_viewer[0][0].set_dynamic_meshes([Mesh(v=true_vertices[0].detach().cpu().numpy(), f=faces)])
                mesh_viewer[0][1].set_dynamic_meshes([Mesh(v=sample_id_vertices[0].detach().cpu().numpy(), f=faces)])
                mesh_viewer[0][2].set_dynamic_meshes([Mesh(v=sample_exp_vertices[0].detach().cpu().numpy(), f=faces)])
                # print(label_same[0])
            elif args.dataset == "BU3DFE":
                faces = np.load(DATASET_PATH + "BU3DFE_face.npy")
                vc = np.ones_like(true_vertices[0].detach().cpu().numpy())
                if id_label_same[0].detach().cpu().numpy() == 0 and exp_label_same[0].detach().cpu().numpy() == 1:
                    vc[:, 1] = 1
                    vc[:, 0] = 0
                    vc[:, 2] = 0
                    mesh_viewer[0][1].set_dynamic_meshes(
                        [Mesh(v=sample_id_vertices[0].detach().cpu().numpy(), f=faces, vc=vc)])
                    mesh_viewer[0][2].set_dynamic_meshes(
                        [Mesh(v=sample_exp_vertices[0].detach().cpu().numpy(), f=faces)])
                elif id_label_same[0].detach().cpu().numpy() == 1 and exp_label_same[0].detach().cpu().numpy() == 0:
                    vc[:, 2] = 1
                    vc[:, 0] = 0
                    vc[:, 1] = 0
                    mesh_viewer[0][1].set_dynamic_meshes(
                        [Mesh(v=sample_id_vertices[0].detach().cpu().numpy(), f=faces)])
                    mesh_viewer[0][2].set_dynamic_meshes(
                        [Mesh(v=sample_exp_vertices[0].detach().cpu().numpy(), f=faces, vc=vc)])
                elif id_label_same[0].detach().cpu().numpy() == 0 and exp_label_same[0].detach().cpu().numpy() == 0:
                    vc[:, 2] = 0
                    vc[:, 0] = 1
                    vc[:, 1] = 0
                    mesh_viewer[0][1].set_dynamic_meshes(
                        [Mesh(v=sample_id_vertices[0].detach().cpu().numpy(), f=faces, vc=vc)])
                    mesh_viewer[0][2].set_dynamic_meshes(
                        [Mesh(v=sample_exp_vertices[0].detach().cpu().numpy(), f=faces, vc=vc)])
                else:
                    mesh_viewer[0][1].set_dynamic_meshes(
                        [Mesh(v=sample_id_vertices[0].detach().cpu().numpy(), f=faces)])
                    mesh_viewer[0][2].set_dynamic_meshes([Mesh(v=sample_exp_vertices[0].detach().cpu().numpy(), f=faces)])
                mesh_viewer[0][0].set_dynamic_meshes([Mesh(v=true_vertices[0].detach().cpu().numpy(), f=faces)])
                # mesh_viewer[0][1].set_dynamic_meshes([Mesh(v=sample_id_vertices[0].detach().cpu().numpy(), f=faces, vc=vc)])
                # mesh_viewer[0][2].set_dynamic_meshes([Mesh(v=sample_exp_vertices[0].detach().cpu().numpy(), f=faces, vc=vc)])
                # sleep(2)
            elif args.dataset == "FaceScape":
                faces = np.load(DATASET_PATH + "facescape_faces.npy")
                mesh_viewer[0][0].set_dynamic_meshes([Mesh(v=true_vertices[0].detach().cpu().numpy(), f=faces)])
                mesh_viewer[0][1].set_dynamic_meshes([Mesh(v=sample_id_vertices[0].detach().cpu().numpy(), f=faces)])
                mesh_viewer[0][2].set_dynamic_meshes([Mesh(v=sample_exp_vertices[0].detach().cpu().numpy(), f=faces)])

            else:
                raise ValueError("No such dataset.")

            # Concatenate true vertices and sampled vertices.
            concatenate_vertices = torch.cat([true_vertices, sample_vertices], dim=2)

            # Forward pass.
            optimiser.zero_grad()
            pred_log_softmax = discriminator(concatenate_vertices)
            loss = F.cross_entropy(pred_log_softmax, label_same)

            # Backward pass.
            loss.backward()
            optimiser.step()
            pred_choice = pred_log_softmax.max(1)[1]
            correct = pred_choice.eq(label_same.data).cpu().sum()

            # Log train losses.
            training_logger.log_batch_loss("loss", loss.item(), "train", pred_log_softmax.shape[0])
            training_logger.log_batch_loss("Accuracy",
                                           correct.item() / pred_log_softmax.shape[0],
                                           "train", pred_log_softmax.shape[0])

        """ Eval """
        discriminator.eval()
        # Evaluate 5 times for stabler results.
        for i in range(5):
            for data in tqdm(test_loader):
                with torch.no_grad():
                    if args.dataset == "BU3DFE":
                        true_vertices, _, _, _, _, _, \
                        sample_id_vertices, id_label_same, sample_exp_vertices, exp_label_same, _, _ = data
                    elif args.dataset == "COMA":
                        true_vertices, _, _, _, _, _, sample_id_vertices, id_label_same, sample_exp_vertices, exp_label_same = data
                    elif args.dataset == "FaceScape":
                        true_vertices, _, _, sample_id_vertices, id_label_same, sample_exp_vertices, exp_label_same, _, _ = data
                    else:
                        raise ValueError("No such dataset.")
                    sample_vertices = sample_id_vertices
                    label_same = id_label_same

                    concatenate_vertices = torch.cat([true_vertices, sample_vertices], dim=2)
                    # Forward pass.
                    pred_log_softmax = discriminator(concatenate_vertices)
                    loss = F.cross_entropy(pred_log_softmax, label_same)

                    pred_choice = pred_log_softmax.max(1)[1]
                    correct = pred_choice.eq(label_same.data).cpu().sum()

                    # Log eval losses.
                    training_logger.log_batch_loss("loss", loss.item(), "eval", pred_log_softmax.shape[0])
                    training_logger.log_batch_loss("Accuracy",
                                                   correct.item() / pred_log_softmax.shape[0],
                                                   "eval", pred_log_softmax.shape[0])
        discriminator.train()

        """ Epoch ends. """
        training_logger.log_epoch({"epoch": epoch,
                                   "model_weights": discriminator.state_dict(),
                                   "optimiser_weights": optimiser.state_dict()})
        if epoch % args.epochs == 0:
            model_save_specific_epochs({"epoch": epoch,
                                        "model_weights": discriminator.state_dict()})
        epoch_end_time = time()
        logging.info("Time left: " + time_format((epoch_end_time - epoch_start_time) * (args.epochs - epoch)))

    full_end_time = time()
    logging.info("Total training time: " + time_format(full_end_time - full_start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain Discriminator.")

    parser.add_argument("-n", "--name", type=str, default="Temporary Experiment",
                        help="Name of the experiment.")
    parser.add_argument("-d", "--dataset", type=str, default="BU3DFE",
                        help="Datasets: BU3DFE or COMA or FaceScape.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--input_channel", type=int, default=6, metavar="N",
                        help="Input channel.")
    parser.add_argument("-e", "--epochs", type=int, default=50, metavar="N",
                        help="Number of episode to train.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, metavar="N",
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", type=int, default=16, metavar="N",
                        help="Batch size for evaluating.")
    parser.add_argument("-k, --k_classes", type=int, default=2, metavar="N",
                        help="Number of the classes.")
    parser.add_argument("-p", "--p_dropout", type=float, default=0.5, metavar="N",
                        help="P for dropout.")
    parser.add_argument("--use_bn", type=bool, default=True,
                        help="Using bn and dropout in the ID discriminator network.")
    parser.add_argument("--info_bn", type=bool, default=False,
                        help="Using information bottleneck.")

    args = parser.parse_args()

    """ args override zone starts. """
    if args.dataset == "COMA":
        args.name = "ID_discriminator_COMA"
        args.epochs = 50
        args.batch_size = 32
    elif args.dataset == "BU3DFE":
        args.name = "ID_discriminator_BU3DFE_10f"
        args.epochs = 100
        args.batch_size = 32
    elif args.dataset == "FaceScape":
        args.name = "ID_discriminator_FaceScape_70percent"
        args.epochs = 100
        args.batch_size = 4
    """ args override zone ends. """

    log_name_dir = LOGS_PATH + args.name + "/"
    if os.path.exists(log_name_dir):
        if args.name == "Temporary Experiment":
            shutil.rmtree(log_name_dir)
            shutil.rmtree(LOGS_PATH + "tf_board/Temporary Experiment")
        else:
            raise ValueError("Name has been used.")
    os.mkdir(log_name_dir)

    logging_level = logging.INFO

    console_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=log_name_dir + "log.txt"), console_handler],
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
    console_handler.setLevel(logging_level)

    logging.info(vars(args))

    training_logger = TrainingLogger(log_name_dir, args)

    discriminator_train(args, training_logger)


