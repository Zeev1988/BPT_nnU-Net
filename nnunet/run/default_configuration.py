#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import nnunet
from nnunet.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.summarize_plans import summarize_plans
from nnunet.training.model_restore import recursive_find_python_class
import numpy as np


def get_configuration_from_output_folder(folder):
    # split off network_training_output_dir
    folder = folder[len(network_training_output_dir):]
    if folder.startswith(os.path.sep):
        folder = folder[len(os.path.sep):]

    configuration, task, trainer_and_plans_identifier = os.path.normpath(folder).split(os.path.sep)
    trainer, plans_identifier = trainer_and_plans_identifier.split("__")
    return configuration, task, trainer, plans_identifier


def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in=(nnunet.__path__[0], "training", "network_training"),
                              base_module='nnunet.training.network_training'):
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'3d\', \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"

    dataset_directory = join(preprocessing_output_dir, task)
    os.makedirs(dataset_directory, exist_ok=True)
    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")

    task_name = 'Task004_Hippocampus'

    plans = load_pickle(plans_file)
    # ##todo: take this out to preprocess or separate function!
    # plans_fname = join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_plans_3D.pkl')
    # plans = load_pickle(plans_fname)
    # plans['plans_per_stage'][0]['batch_size'] = 12
    # plans['plans_per_stage'][0]['patch_size'] = [patch_trial]
    # print("Updated patch size to: ", patch_trial)
    # plans['plans_per_stage'][0]['num_pool_per_axis'] = [7, 7]
    # # because we changed the num_pool_per_axis, we need to change conv_kernel_sizes and pool_op_kernel_sizes as well!
    # plans['plans_per_stage'][0]['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    # plans['plans_per_stage'][0]['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    # # for a network with num_pool_per_axis [7,7] the correct length of pool kernel sizes is 7 and the length of conv
    # # kernel sizes is 8! Note that you can also change these numbers if you believe it makes sense. A pool kernel size
    # # of 1 will result in no pooling along that axis, a kernel size of 3 will reduce the size of the feature map
    # # representations by factor 3 instead of 2.

    # save the plans under a new plans name. Note that the new plans file must end with _plans_2D.pkl!
    # save_pickle(plans, join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_plans_3D.pkl'))

    possible_stages = list(plans['plans_per_stage'].keys())

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    trainer_class = recursive_find_python_class([join(*search_in)], network_trainer,
                                                current_module=base_module)

    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)

    print("###############################################")
    print("I am running the following nnUNet: %s" % network)
    print("My trainer class is: ", trainer_class)
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file)
    print("I am using stage %d from these plans" % stage)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")

    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    print("###############################################")
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class
