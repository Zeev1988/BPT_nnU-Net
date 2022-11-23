import pandas as pd
import os
from tqdm import tqdm
import gc

from BPT import dcm_tool
from BPT import registration
from BPT.bpt_params import BptParams
from BPT.data_utils import BptUtils
from BPT import bet_agent


BET_SCANS_DIR = 'BET_SCANS'
REG_SCANS_DIR = 'REG_SCANS'
RAW_SCANS_DIR = 'RAW_SCANS'

class BrainPreProcessingTool:

    def __init__(self, csv_dir, reg_fixed_mod, bet_fixed_mod, out_dir):
        self.params = BptParams(csv_dir, reg_fixed_mod, bet_fixed_mod, out_dir)
        self.utils = BptUtils(self.params)
        # if external_settings_dict is not None and bool(external_settings_dict):
        #     self.params.set_params_from_dict(external_settings_dict)

    def preprocess(self):
        """
        This is the main function for brain pre processing.
        """
        save_dcm_output = self.params.dcm_save_out
        save_reg_output = self.params.perform_reg and self.params.reg_save_out
        save_bet_output = self.params.perform_bet and self.params.bet_save_out
        df = pd.read_csv(self.params.csv_path, dtype=str).fillna('')
        data = self.utils.df2data(df)
        if BET_SCANS_DIR in data:
            del data[BET_SCANS_DIR]

        reg_summary = []
        bet_summary = []

        bet_out_dir = os.path.join(self.params.out_path, BET_SCANS_DIR)
        reg_out_dir = os.path.join(self.params.out_path, REG_SCANS_DIR)
        raw_out_dir = os.path.join(self.params.out_path, RAW_SCANS_DIR)

      #  resize_dir = self.params.out_path
        bet_dir = bet_out_dir #bet_out_dir if save_bet_output else resize_dir
        reg_dir = reg_out_dir if save_reg_output else bet_dir
        raw_dir = raw_out_dir if save_dcm_output else reg_dir

        for subject_name, studies in tqdm(data.items(), desc=f'Running brain pre processing'):
            for study in studies:
                try:
                    print(f'\nRunning the pre processing tool for patient: {subject_name}, '
                          f'series: {str(list(study.keys())[0])}\n')
                    subject = (subject_name, study)
                    study_id = str(list(study.keys())[0])

                    # Check if any of the output directories exist, and if overwrite mode is not enabled, the case is
                    # skipped
                    if not self.params.overwrite and (os.path.exists(os.path.join(raw_dir, subject_name, study_id)) or
                                                      os.path.exists(os.path.join(reg_dir, subject_name, study_id)) or
                                                      os.path.exists(os.path.join(bet_dir, subject_name, study_id))):
                        continue

                    # convert dicom files to .nii/.nii.gz format
                    out_dir = os.path.join(raw_dir, subject_name, study_id)
                    dcm_tool.dcm2nii(subject, out_dir, self.params.shrink_output, self.params.modalities)
                    self.utils.rename_copy_files(subject, out_dir, self.params.modalities)
                    if save_dcm_output:
                        raw_summary.append(self.utils.generate_data(subject))

                    # co-register images
                    if self.params.perform_reg:
                        out_dir = os.path.join(reg_dir, subject_name, study_id)
                        registration.register(subject, self.params.elastix_exe, self.params.elastix_params, out_dir,
                                              self.params.modalities, self.params.reg_fixed_module, self.params.label_name,
                                              self.params.shrink_output)
                        if save_reg_output:
                            reg_summary.append(self.utils.generate_data(subject))

                    # BET
                    if self.params.perform_bet:
                        out_dir = os.path.join(bet_dir, subject_name, study_id)
                        bet_agent.bet(subject, out_dir, self.params.bet_fixed_module, self.params.perform_n4,
                                      self.params.shrink_output, self.params.modalities, self.params.label_name)
                    bet_summary.append(self.utils.generate_data(subject))
                except:
                    print(f'{subject_name} - Failed BET')

            gc.collect()

        if save_dcm_output:
            self.utils.data2csv(raw_summary, raw_dir)
        if save_reg_output:
            self.utils.data2csv(reg_summary, reg_dir)
        if save_bet_output:
            self.utils.data2csv(bet_summary, bet_dir)

        return bet_dir
# if __name__ == '__main__':
#     BrainPreProcessingTool().preprocess()
