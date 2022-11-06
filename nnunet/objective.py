import optuna
import shutil
from nnunet.training.network_training import nnUNetTrainerV2
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances
import nnunet.run.run_training as rt
from batchgenerators.utilities.file_and_folder_operations import *
import subprocess as sp

OPTUNA = "Optuna (all)"
ALL_OPTIMIZERS = ["Adam", "SGD"]
ALL_LOSSES = ["DiceCE", "RobustCrossEntropyLoss", "Dice_Focal", "Dice_TopK10", "Dice_TopK10_CE", "Dice_TopK10_Focal"]

USED_PARAMS = {}


def callback(study, trial):
    save_model_and_graph(str(trial.params), True)

def save_model_and_graph(params, optuna):
    path_src = join(OUTPUT_FOLDER, "model_final_checkpoint.model")
    path_src_pkl = join(OUTPUT_FOLDER, "model_final_checkpoint.model.pkl")
    path_src_img = join(OUTPUT_FOLDER, "progress.png")
    print("trial param", params, str(params))
    param_dict = eval(params)
    if optuna:
        params_string = ""
        if 'Optimizer' in params:
            params_string +=  param_dict['Optimizer']
        if 'LossFunction' in params:
            params_string += param_dict['LossFunction']
        if 'Epochs' in params:
            params_string += param_dict['Epochs']
    else:
        params_string = param_dict['Optimizer'] + "_" + param_dict['LossFunction'] + "_" + param_dict['Epochs']
    path_dst = join(OUTPUT_FOLDER, "model_final_checkpoint_%s.model" % params_string)
    path_dst_pkl = join(OUTPUT_FOLDER, "model_final_checkpoint_%s.model.pkl" % params_string)
    path_dst_img = join(OUTPUT_FOLDER, "progress_%s.png" % params_string)
    shutil.copy2(path_src, path_dst)
    shutil.copy2(path_src_pkl, path_dst_pkl)
    shutil.copy2(path_src_img, path_dst_img)


def objective(trial, data_set, configuration, transfer_learning, min_epochs, max_epochs, optimizer, loss_function, fold):
    print("in objective")
    Params = choose_trial_params(trial, data_set, configuration, transfer_learning, min_epochs, max_epochs, optimizer,
                      loss_function, fold)

    with open("logs.txt", 'a') as log_file:
        log_file.write("########################################################\n")
        log_file.write("Starting Trial %d \n" % trial.number)
        log_file.write("With parameters: %s \n" % str(Params))

    trainer = rt.main(Params)

    dice_score = trainer.all_val_eval_metrics[-1]
    global OUTPUT_FOLDER
    OUTPUT_FOLDER = trainer.output_folder
    print("output folder 2 ", OUTPUT_FOLDER)

    with open("logs.txt", 'a') as log_file:
        log_file.write("Trial Score: %f with parameters: %s\n" % (dice_score, str(Params)))

    return dice_score


def choose_trial_params(trial, data_set, configuration, transfer_learning, min_epochs, max_epochs, optimizer, loss_function, fold):
    Optimizer = optimizer
    LossFunction = loss_function
    if optimizer == OPTUNA:
        Optimizer = trial.suggest_categorical("Optimizer", ALL_OPTIMIZERS)
    if loss_function == OPTUNA:
        LossFunction = trial.suggest_categorical("LossFunction", ALL_LOSSES)
    if min_epochs != max_epochs:
        Epochs = trial.suggest_int("Epochs", int(min_epochs), int(max_epochs))
        print("EPOCHS " + str(min_epochs) + str(max_epochs))
    else:
        Epochs = int(min_epochs)
        print("EPOCHS " + str(min_epochs))

    Params = {"Optimizer": Optimizer, "LossFunction": LossFunction, "Epochs": Epochs, "Transfer": transfer_learning, "DataSet": data_set, "Configuration": configuration, "Fold": fold}
    return Params


def start_nnunet(data_set, configuration, transfer_learning, epochs, optimizer, loss_function, folds):
    programName = r"C:\Program Files\Notepad++\notepad++.exe"
    fileName = r"D:\users\zeevh\nnUNet\nnUNet-1-New\nnunet\logs.txt"

    sp.Popen([programName, fileName])
    for f in folds:
        Params = {"Optimizer": optimizer, "LossFunction": loss_function, "Epochs": int(epochs),
                  "Transfer": transfer_learning,
                  "DataSet": data_set, "Configuration": configuration, "Fold": f}
        with open("logs.txt", 'w') as log_file:
            log_file.write("Running nnUNet.. \n")
            log_file.write("Parameters: %s \n" % str(Params))
            log_file.write("########################################################\n")

        trainer = rt.main(Params)
        dice_score = trainer.all_val_eval_metrics[-1]

        with open("logs.txt", 'a') as log_file:
            log_file.write("########################################################\n")
            log_file.write("Fold %d Dice Score: %f \n" %(int(f),dice_score))
            log_file.write("Model is: model_final_checkpoint")

    return "Final Dice Score: FIX THIS!!!!"


def start_optuna(data_set, configuration, transfer_learning, max_epochs, optimizer, loss_function, folds):
    programName = r"C:\Program Files\Notepad++\notepad++.exe"
    fileName = r"D:\users\zeevh\nnUNet\nnUNet-1-New\nnunet\logs.txt"
    with open("logs.txt", 'w') as log_file:
        log_file.write("Running nnUNet with Optuna optimization.. \n")
    sp.Popen([programName, fileName])
    for f in folds:
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), direction='maximize')
        study.optimize(lambda trial: objective(trial, data_set, configuration, transfer_learning, int(max_epochs), optimizer, loss_function, f), n_trials=20, timeout=None, show_progress_bar=True, callbacks = [callback])
        print_trial_deatils(study)
        add_optuna_visualization(study, f)
        with open("logs.txt", 'a') as log_file:
            log_file.write("########################################################\n")
    res = ""
    model_suffix = ""
    #todo: fix prints
    #for key, value in study.best_params.items():
    #    res += str(key) + ": " + str(value) + "\n"
     #   model_suffix+= "_" + str(value)
    #print(model_suffix)
    #summary_string = "Best parameters: \n" + res + "Final Dice Score:" + study.best_value +"\n" + "Best model is: model_final_checkpoint" + model_suffix
    #print(summary_string)
        #log_file.write(summary_string)
   # return summary_string

def print_trial_deatils(study):
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")

    with open("logs.txt", 'a') as log_file:
        log_file.write("########################################################\n")
        log_file.write("Best is trial %d with score %f\n" % (study.best_trial.number, study.best_trial.value))
        log_file.write("Best parameters: \n")
        for key, value in study.best_params.items():
            log_file.write(f"\t{key}: {value} \n")

def add_optuna_visualization(study,fold):
    print("fold %d visualization is available"%(fold), optuna.visualization.is_available())
    optuna.visualization.plot_optimization_history(study).show(renderer="browser")
    optuna.visualization.plot_param_importances(study).show(renderer="browser")
    fig_param_importances = optuna.visualization.plot_param_importances(study)
    fig_param_importances.write_image('image_param_importances.png')



#if __name__ == "__main__":
 #   main()

