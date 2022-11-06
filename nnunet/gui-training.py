import gradio as gr
from objective import *
from nnunet.experiment_planning.nnUNet_plan_and_preprocess import preprocess
import sys

import time
OPTUNA = "Optuna (all)"
LOSS_FUNCTIONS_MAP = {"Optuna (all)": "Optuna (all)", "Dice CE": "DiceCE", "SoftDice CE": "SoftDiceCE", "Robust Cross-Entropy Loss": "RobustCrossEntropyLoss",
                      "Dice Focal": "Dice_Focal", "Dice Top-K 10": "Dice_TopK10",
                      "Dice Top-K 10 CE": "Dice_TopK10_CE", "Dice Top-K 10 Focal": "Dice_TopK10_Focal"}

def greet(data_set, configuration, do_transfer_lr, do_preproccesing, max_epochs, optimizer, loss_function, folds):
    if do_preproccesing:
        preprocess([int(data_set)])
    try:
        if optimizer != OPTUNA and loss_function != OPTUNA:
            res = start_nnunet(data_set, configuration, do_transfer_lr, max_epochs, optimizer,
                               LOSS_FUNCTIONS_MAP[loss_function], folds)
        else:
            res = start_optuna(data_set, configuration, do_transfer_lr, max_epochs, optimizer,
                               LOSS_FUNCTIONS_MAP[loss_function], folds)
    except Exception as inst:
        with open(r"D:\users\zeevh\nnUNet\nnUNet-1-New\nnunet\logs.txt", 'a') as log_file:
            log_file.write("Error: %s\n" % inst)  ##print_epoch_to_logfile
        return "Error: %s\n" % inst
    return res

def main():
    choices_optimizer = [OPTUNA, "Adam", "SGD"]
    choices_loss_function = [OPTUNA,  "Dice CE", "SoftDice CE", "Robust Cross-Entropy Loss", "Dice Focal", "Dice Top-K 10", "Dice Top-K 10 CE", "Dice Top-K 10 Focal"]
    choices_fols = ['0','1','2','3','4']
    description = "Choose the data set, configuration and transfer learning (tick to use). For optimizer and loss function choose fixed value or Optuna optimization (will be done over all values). For epochs, choose min and max values to provide a range for Optuna to optimize over. For fixed value, write it for both min and max."
    optimizer = gr.inputs.Radio(choices_optimizer, default=[], type="value", label=None)
    loss_function = gr.inputs.Radio(choices_loss_function, default=[], type="value", label=None)
    folds = gr.inputs.CheckboxGroup(choices_fols, default=[], type="value", label=None)
    max_epochs = gr.inputs.Textbox(lines=1, placeholder=None, default="10", numeric=False, type="number", label="epochs - maximum value")
    do_preproccesing = "checkbox"
    do_transfer_lr = "checkbox"
    configuration = gr.inputs.Dropdown(['2d', '3d_fullres', '3d_cascade_fullres', '3d_lowers'])
    data_set = gr.inputs.Textbox(lines=1, placeholder="task number (E.g. 2)", default=None, numeric=False, label="dataset")
    folds
    iface = gr.Interface(
        fn=greet,
        inputs=[data_set, configuration, do_transfer_lr, do_preproccesing, max_epochs, optimizer, loss_function, folds],
        outputs=["text"],
        layout = "vertical",
        title = "Training",
        description = description,
        css = "design-gui-training.css",
        allow_screenshot = False,
        allow_flagging = False)
    iface.launch(inbrowser = True)
    return

if __name__ == "__main__":
    main()