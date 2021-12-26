import os
from torch import device, cuda
from torch.backends.cudnn import benchmark


device = device("cuda" if cuda.is_available() else "cpu")
benchmark = True

PROJECT_PATH = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)) + '/'
DATASET_PATH = PROJECT_PATH + "data/"
BU3DFE_PATH = DATASET_PATH + "BU3DFE/"
COMA_PATH = DATASET_PATH + "COMA/"
LOGS_PATH = PROJECT_PATH + "logs/"
EVAL_PATH = PROJECT_PATH + "Evaluations/"
FACESCAPE_PATH = DATASET_PATH + "FaceScape/"

BU3DFE_NORMALISE = 156
COMA_NORMALISE = 0.26
FACESCAPE_NORMALISE = 573


def time_format(time_diff):
    hours, rem = divmod(time_diff, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


class Bunch(object):
    def __init__(self, a_dict):
        self.__dict__.update(a_dict)