# Adversarial Face Disentanglement

This repository is an implementation of the IEEE FG 2023 paper 'Adversarial 3D Face Disentanglement of Identity and Expression' by Yajie et al. Full paper is available [here]().

## Requirements
* Python 3.7
* PyTorch 1.9.0
* setuptools 59.5.0
* psbody Mesh (for visualisation)
* numpy
* scipy
* tensorboard
* cv2
* tqdm

## Preparation
### Visualisation (optional)
If visualisation is required, install psbody Mesh at [https://github.com/MPI-IS/mesh](https://github.com/MPI-IS/mesh).

### Dataset preparation
Download all datasets: [BU-3DFE](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html), [CoMA](https://coma.is.tue.mpg.de/index.html) and [FaceScape](https://facescape.nju.edu.cn/Page_Download/), and place under `data/[Dataset Name]/` folder.

For BU-3DFE dataset, place all `*.wrl` files under `data/BU3DFE/`.

For CoMA dataset, place all folders with pattern `FaceTalk_*_*_TA` under `data/CoMA/`.

For FaceScape dataset, place all subjects' folders under `data/FaceScape/`.

### Dataset preprocess
Go into the `utils` folder by `cd utils` (Ignore this if running using an IDE).

For BU-3DFE dataset, run:
```shell
PYTHONPATH=$PYTHONPATH:../ python bu3dfe.py
```
or
```shell
set PYTHONPATH=../
python bu3dfe.py
```

For CoMA dataset, run:
```shell
PYTHONPATH=$PYTHONPATH:../ python coma_preprocess.py
```

For FaceScape dataset, first down-sample an arbitrary mesh, e.g., `1_neutral.obj`. The tool used for this process is MeshLab.

The process is `Filters->Remeshing, Simplification and Reconstruction->Simplification: Quadric Edge Collapse Decimation` with target number of faces set to 9000.

Then save both the original `1_neutral.obj` and the down-sampled `1_neutral_downsampled.obj` under `data/FaceScape/downsample`.

Run:
```shell
PYTHONPATH=$PYTHONPATH:../ python find_facescape_indices_simp.py
PYTHONPATH=$PYTHONPATH:../ python facescape_prepocess.py
```

## Pretrain
Go back to the project folder by `cd ..` (Ignore this if running using an IDE).

Run pretrain for each dataset:
```shell
python pretrain_discriminator.py --dataset COMA
python pretrain_discriminator.py --dataset BU3DFE
python pretrain_discriminator.py --dataset FaceScape
```

## Train
Run train for each dataset:

- using neutral ground-truths (using id discriminator is optional)
```shell
python end_to_end.py --dataset COMA --name "COMA_withgt_withiddis" --with_neutral_gt True --latent_vector_dim_id 4 --latent_vector_dim_exp 4 --lr_id_discriminator 1e-3 --lambda1 5000 --lambda2 3e-3 --lambda3 3e-3 --lambda4 1e-5 --lambda6 5 --lambda7 7e-3 --lambda8 7e-3 --use_bn True --epochs 300 
python end_to_end.py --dataset BU3DFE --name "BU3DFE_withgt_withiddis" --with_neutral_gt True --latent_vector_dim_id 40 --latent_vector_dim_exp 40 --lambda1 250 --lambda2 3e-5 --lambda3 3e-5 --lambda6 0.5 --lambda7 5e-2 --lambda8 10 --batch_size 8 --epochs 280
python end_to_end.py --dataset FaceScape --name "FaceScape_withgt_withiddis" --with_neutral_gt True --latent_vector_dim_id 64 --latent_vector_dim_exp 64 --lambda1 5000 --lambda2 3e-5 --lambda3 3e-5 --lambda4 5e-4 --lambda6 1 --epochs 280
```
- not using neutral ground-truths (using id discriminator is optional)
```shell
python end_to_end.py --dataset COMA --name "COMA_wogt_withiddis" --latent_vector_dim_id 4 --latent_vector_dim_exp 4 --lambda1 600 --lambda4 5e-4 --lambda7 5e-3 --epochs 300
python end_to_end.py --dataset BU3DFE --name "BU3DFE_wogt_withiddis" --latent_vector_dim_id 40 --latent_vector_dim_exp 40 --lambda1 250 --lambda2 3e-5 --lambda3 3e-5 --lambda6 0.5 --lambda7 5e-2 --lambda8 10 --batch_size 8 --epochs 280
python end_to_end.py --dataset FaceScape --name "FaceScape_wogt_withiddis" --latent_vector_dim_id 64 --latent_vector_dim_exp 64 --lambda1 5000 --lambda2 3e-5 --lambda3 3e-5 --lambda4 5e-4 --lambda6 1 --epochs 280
```

## Evaluate
Run evaluate for each dataset:
```shell
cd Model
PYTHONPATH=$PYTHONPATH:../ python evaluation_vae.py BU3DFE_withgt_withiddis
PYTHONPATH=$PYTHONPATH:../ python evaluation_vae.py COMA_withgt_withiddis
PYTHONPATH=$PYTHONPATH:../ python evaluation_vae.py FaceScape_withgt_withiddis
```

It will show reconstruction meshes (if psbody.mesh is installed, otherwise relevant code has to be commented), and at the end of each evaluation, reconstruction errors and disentanglement error will be shown.


## Applications

### Expression transfer and interpolation
For CoMA and FaceScape, we show applications on expression transfer and identity and expression interpolation.

Run expression transfer and interpolation for CoMA and FaceScape (still in the `Model` folder):
```shell
PYTHONPATH=$PYTHONPATH:../ python exp_trans_inter.py COMA_withgt_withiddis
PYTHONPATH=$PYTHONPATH:../ python exp_trans_inter.py FaceScape_withgt_withiddis
```
The shown samples are saved under `logs/exp_interpolate` and `logs/exp_transfer`.

### Face Recognition
For BU-3DFE and FaceScape, we show the rank-1 accuracy of face recognition.

Run face recognition for BU-3DFE and FaceScape (still in the `Model` folder):
```shell
PYTHONPATH=$PYTHONPATH:../ python face_recognition.py BU3DFE_withgt_withiddis
PYTHONPATH=$PYTHONPATH:../ python face_recognition.py FaceScape_withgt_withiddis
```


## Cite the work
```
```


## Acknowledgements
This code repo is based on [PointNet](https://github.com/charlesq34/pointnet). We thank the authors for their great job!


## License
It is released under the MIT License. See the [LICENSE file](https://github.com/rmraaron/FaceExpDisentanglement/blob/main/LICENSE) for more details.
