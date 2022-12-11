# Preprocessing and segmentation tool for MRI images.
This is an automated pipline for for MRI images preprocessing and segmentation(nnU-Net) for medical data.
Trained and tested for multiple sclerosis lesions segmentation

#The preprocessing stage is optional(!) and consists of:
- DICOM to NIfTI conversion (if needed).
- Images registration.
- N4 Bias Field Correction
- Brain extruction (using https://github.com/MIC-DKFZ/HD-BET).

## Requirments
nnU-Net requires a GPU! For inference, the GPU should have 4 GB of VRAM. For training nnU-Net models the GPU should have at 
least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080 or RTX 3090). Due to the use of automated mixed 
precision, fastest training times are achieved with the Volta architecture (Titan V, V100 GPUs) when installing pytorch 
the easy way. Since pytorch comes with cuDNN 7.6.5 and tensor core acceleration on Turing GPUs is not supported for 3D 
convolutions in this version, you will not get the best training speeds on Turing GPUs. You can remedy that by compiling pytorch from source 
(see [here](https://github.com/pytorch/pytorch#from-source)) using cuDNN 8.0.2 or newer. This will unlock Turing GPUs 
(RTX 2080ti, RTX 6000) for automated mixed precision training with 3D convolutions and make the training blistering 
fast as well. Note that future versions of pytorch may include cuDNN 8.0.2 or newer by default and 
compiling from source will not be necessary.

## Installation
### windows
In an anaconda prompt:
```bash
conda create -n bpt__nnunet3.8 python=3.8 (yes to all)
conda activate bpt__nnunet3.8
conda config --set ssl_verify False
cd <path to project>
pip install -r ./requirements.txt (yes to all)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia (yes to all)
```
### Linux
1. Go to https://github.com/SuperElastix/elastix/releases/tag/5.0.0 and download "elastix-5.0.0-linux.tar.bz2" and extract it wherever you like.
2. Open a new terminal and do ```vi ~/.bashrc```
3. Add the following lines to the file:
```bash
export PATH=<path to extracted folder>/elastix-5.0.0-Linux/bin:$PATH
export LD_LIBRARY_PATH=<path to extracted folder>/elastix-5.0.0-Linux/lib:$LD_LIBRARY_PATH
```
4. Close the Terminal.
5. Open a new terminal and do:
```bash
cd <path to project>
virtualenv  bpt__nnunet3.8 -p python3.8
virtualenv -p /usr/bin/python3.8 bpt__nnu-net3.8
source bpt__nnunet3.8/bin/activate
pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org -r ./requirements.txt -vvv
pip install torch torchvision torchaudio 
```

## Prepare the data
Before executing, make sure that the directory you are working on is orgenized as following:

    <FOLDER FOR CASES>/
    ├── <FOLDER SUBJECT 1>
    │   └── <FOLDER STUDY 1>
    │       └── nnUNetTrainerV2__nnUNetPlansv2.1
    │           ├── FLAIR (folder or nii)
    │           ├── T1 (folder or nii)
    │           ├── T1C (folder or nii)
    │           ├── T2 (folder or nii)
    │       └── <FOLDER STUDY 2>
    ├── <FOLDER SUBJECT 2>

This pipline can only work with FLAIR, T1, T1C, T2 modalities. You don't need to have all the modalities. 
Make sure that you have the needed modalities for 
registration (default is FLAIR), brain extruction(default is T1C), and inference (dependence on the model you chooce)! 
or the process will fail!!!

## Run inference
For prediction, user must provide a path to a trained nnU-Net model.

```bash
set CUDA_VISIBLE_DEVICES=<number of gpu>
python <path to project>/gui-test.py 
```

For both preprocessing and segmentation, The output is saved to <path to original cases>/BET_SCANS directory.
