import numpy as np
from bisect import bisect_left
from tqdm import tqdm
from multiprocessing import Pool
import cv2
import torch


def obj_read(filename, texture_filename=None):
    f = open(filename)
    lines = f.readlines()
    vertices = []
    faces = []
    texture_coords = []
    for i in lines:
        if i[0] == "f":
            # Process faces.
            fi = i[2:].split(" ")
            vertex_ind = []
            for j in fi:
                vertex_ind.append(int(j.split("/")[0]) - 1)
            faces.append(vertex_ind)
        elif i[:2] == "v ":
            # Process vertices.
            v = i[2:].strip()
            v = v.split(" ")
            vertices.append([float(vv) for vv in v])
        elif i[:3] == "vt ":
            # Process texture coordinates per vertex.
            vt = i[3:].strip()
            vt = vt.split(" ")
            texture_coords.append([float(vtt) for vtt in vt])
    f.close()

    vertices = np.array(vertices)
    if vertices.shape[1] == 6:
        colours = vertices[:, 3:]
        vertices = vertices[:, :3]
    elif texture_filename is not None and texture_coords:
        texture_image = cv2.imread(texture_filename) / 255.
        texture_image = texture_image[:, :, ::-1]
        texture_width, texture_height = texture_image.shape[0], texture_image.shape[1]
        texture_coords = np.array(texture_coords)
        texture_coords[:, 0] = texture_coords[:, 0] * texture_width
        texture_coords[:, 1] = texture_coords[:, 1] * texture_height
        texture_coords = texture_coords.astype(np.int)
        colours = texture_image[texture_coords[:, 0], texture_coords[:, 1], :]
    else:
        colours = None
    faces = np.array(faces)
    if faces.max() < -1:
        faces = faces + faces.min() * -1
    return vertices, colours, faces


def _obj_write(f, vertices, faces, faces_offset, colours=None, landmarks=None):
    f.write("# OBJ file\n")
    for v_ind, v in enumerate(vertices):
        if colours is not None:
            c = colours[v_ind]
        else:
            c = [1.0, 1.0, 1.0]
        if landmarks is not None:
            if v_ind in landmarks:
                c = [1.0, 0.0, 0.0]

        f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], c[0], c[1], c[2]))

    for p in faces + faces_offset:
        f.write("f %d %d %d\n" % (p[0], p[1], p[2]))


def obj_write(filename, vertices, faces, colours=None, landmarks=None):
    with open(filename, "w") as f:
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        if isinstance(colours, torch.Tensor):
            colours = colours.detach().cpu().numpy()
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.detach().cpu().numpy()
        _obj_write(f, vertices, faces, faces_offset=1, colours=colours, landmarks=landmarks)
        f.close()