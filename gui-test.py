import os
import gradio as gr
import nnunet.inference.predict as inf
import subprocess as sp
from BPT.bpt import BrainPreProcessingTool
from nnunet_bpt_utils import NnunetBptUtils
import tempfile
from sys import platform
import shutil

def predict(model, input_dir, output_dir):
    inf.predict_from_folder(model, input_dir, output_dir, None, False, 6, 2, None, False, 1, True, mixed_precision=True,
                            overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None, step_size=0.5)



def greet(input_path, model, do_bpt, reg, bet):
    nnunet_tmp_dir = os.path.join(tempfile.gettempdir(), 'TMP')
    nnunet_res_dir = os.path.join(tempfile.gettempdir(), 'RES')
    try:        
        assert input_path, "No input"
        nb_utils = NnunetBptUtils()
        nb_utils.dir2csv(input_path)
        out_path = input_path
        if do_bpt:
            bpt = BrainPreProcessingTool(os.path.join(input_path, 'summary.csv', ), reg, bet, input_path)
            out_path = bpt.preprocess()
            nb_utils.dir2csv(out_path)

        if model:
            os.makedirs(nnunet_tmp_dir, exist_ok=True)
            os.makedirs(nnunet_res_dir, exist_ok=True)

            modalities = nb_utils.get_requirements_from_model(model)
            nb_utils.ichilov_to_nnunet_format(modalities, os.path.join(out_path, 'summary.csv'), nnunet_tmp_dir)
            predict(model, nnunet_tmp_dir, nnunet_res_dir)


            nb_utils.pred_to_original_path(os.path.join(nnunet_tmp_dir, 'summary.csv'), nnunet_res_dir, out_path)

        sp.Popen(["explorer" if platform =="win32" else 'xdg-open', out_path])
    except:
        pass
    
    if os.path.exists(nnunet_tmp_dir):
        shutil.rmtree(nnunet_tmp_dir, ignore_errors=True)
    if os.path.exists(nnunet_res_dir):
        shutil.rmtree(nnunet_res_dir, ignore_errors=True)
    
    return


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # MRI brain images preprocessing and inference tool
    """
    )
    input_path = gr.Textbox(lines=1, placeholder=None, label="Insert data location for preprocessing/inference")
    model = gr.Textbox(lines=1, placeholder=None, label="Insert model location for inference")
    bpt = gr.Checkbox(value=False, label="Do preprocessing")

    with gr.Column(visible=False) as details_col:
        reg = gr.Radio(choices=["FLAIR", "T1", "T1C", "T2"], label="Fix contrast for registration")
        bet = gr.Radio(choices=["FLAIR", "T1", "T1C", "T2"], label="Fix contrast for brain extraction")

    generate_btn = gr.Button("Generate")
    output = gr.Textbox(label="Output")


    def radio_groups_visible(bpt):
        return gr.update(visible=bpt)


    bpt.change(radio_groups_visible, bpt, details_col)
    generate_btn.click(greet, [input_path, model, bpt, reg, bet], output)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
