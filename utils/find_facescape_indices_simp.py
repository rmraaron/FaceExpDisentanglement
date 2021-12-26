from psbody.mesh import Mesh
from scipy.spatial import KDTree
import numpy as np
from path import *
import os


if __name__ == '__main__':
    if not os.path.exists(FACESCAPE_PATH+"downsample/"):
        os.mkdir(FACESCAPE_PATH+"downsample/")
    # 1_neutral.obj is an original mesh and 1_neutral_downsampled is the corresponding downsampled mesh
    original = Mesh(filename=FACESCAPE_PATH+"downsample/1_neutral.obj").v
    downsampled = Mesh(filename=FACESCAPE_PATH+"downsample/1_neutral_downsampled.obj")
    idx = KDTree(original).query(downsampled.v)[1]
    Mesh(v=original[idx], f=downsampled.f).write_obj("vis.obj")
    np.save(FACESCAPE_PATH+"downsample/downsampled_v_idx", idx)
    np.save(FACESCAPE_PATH+"downsample/downsampled_f", downsampled.f)
