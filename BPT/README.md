# BPT
A tool for preprocessing of MRI scans

# Installation Instructions(from fabian@Fabian rep)
Note that you need to have a python3 installation for HD-BET to work. Please also make sure to install HD-BET with the correct pip version (the one that is connected to python3). You can verify this using the ```--version``` command:
```pip --version```

Once python 3 and pip are set up correctly, run the following commands to install HD-BET:

Clone this repository:
```
git clone https://github.com/Zeev1988/BPT
```
Go into the repository (the folder with the setup.py file) and install:
```
cd HD-BET
pip install -e .
```
Per default, model parameters will be downloaded to ~/hd-bet_params. If you wish to use a different folder, open bpt_params.py in a text editor and modify folder_with_parameter_files

# How to use it
