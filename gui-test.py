import os
import shutil

import gradio as gr
import nnunet.inference.predict as inf
import subprocess as sp
from BPT.bpt import BrainPreProcessingTool
from nnunet_bpt_utils import NnunetBptUtils
import tempfile

def predict(model, input_dir, output_dir):
    inf.predict_from_folder(model, input_dir, output_dir, None, False, 6, 2, None, False, 1, True, mixed_precision=True,
                            overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None, step_size=0.5)


def delete_tmp_dirs(nnunet_tmp_dir, nnunet_res_dir):
    if os.path.exists(nnunet_tmp_dir):
        shutil.rmtree(nnunet_tmp_dir)

    if os.path.exists(nnunet_res_dir):
        shutil.rmtree(nnunet_res_dir)


def greet(input_path, model, do_bpt):
    input_path = r'D:\users\zeevh\VOL\nnU-Net_data\original_data\TASMC\aaa'
    model = r'D:\users\zeevh\nnUNet\nnUNet_trained_models\nnUNet\3d_fullres\Task000_ISBI_BASE_UNI\nnUNetTrainerV2__nnUNetPlansv2.1'
    nb_utils = NnunetBptUtils()
    nb_utils.dir2csv(input_path)
    out_path = input_path
    if do_bpt:
        bpt = BrainPreProcessingTool(os.path.join(input_path, 'summary.csv', ), input_path)
        out_path = bpt.preprocess()
        nb_utils.dir2csv(out_path)

    if model:
        nnunet_tmp_dir = os.path.join(out_path, tempfile.gettempdir(), 'TMP')
        os.makedirs(nnunet_tmp_dir, exist_ok=True)

        nnunet_res_dir = os.path.join(out_path, tempfile.gettempdir(), 'RES')
        os.makedirs(nnunet_res_dir, exist_ok=True)

        modalities = nb_utils.get_requirements_from_model(model)
        nb_utils.ichilov_to_nnunet_format(modalities, os.path.join(out_path, 'summary.csv'), nnunet_tmp_dir)
        predict(model, nnunet_tmp_dir, nnunet_res_dir)


        nb_utils.pred_to_original_path(os.path.join(nnunet_tmp_dir, 'summary.csv'), nnunet_res_dir, out_path)
        delete_tmp_dirs(nnunet_tmp_dir, nnunet_res_dir)
        sp.Popen(["explorer", out_path])
    return

def main():
    input_path = gr.inputs.Textbox(lines=1, placeholder=None, numeric=False, type="str", label=None)
    model = gr.inputs.Textbox(lines=1, placeholder=None, numeric=False, type="str", label=None)
    do_bpt = "checkbox"

    description = "Insert input folder with the images to run inference on, and output folder to save the labels produced. The inference will use the best model which was produced in training on the data set and configuration you state here."
    iface = gr.Interface(
        fn=greet,
        inputs=[input_path, model, do_bpt],
        outputs=["text"],
        layout = "vertical",
        title = "Inference",
        description = description,
        allow_screenshot = False,
        allow_flagging = False)
    iface.launch(inbrowser = True)

if __name__ == "__main__":
    main()