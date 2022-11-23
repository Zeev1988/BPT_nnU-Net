import csv
import pickle

import pandas as pd
import os
import pathlib
import re
import shutil
from BPT.bpt_params import BptParams
from nnunt_bpt_params import NnunetBbtParams


# directory structure:
# - parent dir
# -- subject dir
# --- study dir
# ---- FLAIR DIR or nii
# ---- T1C DIR or nii
# ---- T1 DIR or nii
# ---- T2 DIR or nii


class NnunetBptUtils:
    def __init__(self):
        self.params = BptParams()
        self.nnunet = NnunetBbtParams()

    def dir2csv(self, path):
        header = [self.params.subject_col_name, self.params.study_col_name,
                  self.params.t1_name, self.params.t1ce_name, self.params.t2_name, self.params.flair_name,
                  self.params.label_name]
        subjects = []
        studies = []
        t1s = []
        t1cs = []
        t2s = []
        flairs = []
        labels = []
        for subject in os.listdir(path):
            try:
                subj_path = os.path.join(path, subject)
                if os.path.isdir(subj_path):
                    for study in os.listdir(subj_path):
                        stud_path = os.path.join(path, subject, study)
                        if os.path.isdir(stud_path):
                            files_list = os.listdir(stud_path)
                            flair = [i for i in files_list if self.params.flair_name in i]
                            flair = os.path.join(stud_path, flair[0]) if len(flair) else ""

                            t2 = [i for i in files_list if self.params.t2_name in i]
                            t2 = os.path.join(stud_path, t2[0]) if len(t2) else ""

                            label = [i for i in files_list if self.params.label_name in i]
                            label = os.path.join(stud_path, label[0]) if len(label) else ""

                            t1c = [i for i in files_list if self.params.t1ce_name in i]
                            t1c = os.path.join(stud_path, t1c[0]) if len(t1c) else ""

                            t1 = [i for i in files_list if
                                  (self.params.t1_name in i) and (self.params.t1ce_name not in i)]
                            t1 = os.path.join(stud_path, t1[0]) if len(t1) else ""

                            subjects.append(subject)
                            studies.append(study)
                            flairs.append(flair)
                            t1s.append(t1)
                            t1cs.append(t1c)
                            t2s.append(t2)
                            labels.append(label)
            except:
                print(f'dir2csv - Failed to process subject - {subject}')

        with open(os.path.join(path, 'summary.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(subjects, studies, t1s, t1cs, t2s, flairs, labels))

    def pred_to_original_path(self, summary_path, res_path, original_folder):
        df = pd.read_csv(summary_path, na_filter='')
        conv_dict = {row[self.nnunet.subject_column]: row[self.nnunet.original_column] for _, row in df.iterrows()}
        results = [f for f in os.listdir(res_path) if f.endswith("nii.gz")]
        for res in results:
            lbl_path = os.path.join(res_path, res)
            ext = ''.join(pathlib.Path(lbl_path).suffixes)
            shutil.copy(lbl_path,
                        os.path.join(original_folder, conv_dict[int(re.findall("(\d+)(?!.*\d)", lbl_path)[0])],
                                     f'prediction{ext}'))


    def get_requirements_from_model(self, model_path):
        with open(os.path.join(model_path, 'plans.pkl'), 'rb') as f:
            data = pickle.load(f)
            modalities = data['modalities']
            return {y: x for x, y in modalities.items()}


    def nnunet_modality_to_bpt_modality(self, mod):
        if mod == self.nnunet.flair:
            return self.params.flair_name
        if mod == self.nnunet.t1:
            return self.params.t1_name
        if mod == self.nnunet.t1ce:
            return self.params.t1ce_name
        if mod == self.nnunet.t2:
            return self.params.t2_name


    def ichilov_to_nnunet_format(self, modalities, summary_path, out_scans_path):
        df = pd.read_csv(summary_path, na_filter='')
        summary = []
        for index, row in df.iterrows():
            if all([len(row[self.nnunet_modality_to_bpt_modality(key)])for key in modalities.keys()]):
                for modality, id in modalities.items():
                    col = self.nnunet_modality_to_bpt_modality(modality)
                    ext = ''.join(pathlib.Path(row[col]).suffixes)
                    shutil.copy(row[col], os.path.join(out_scans_path,
                                                        f'{self.nnunet.prefix}_{index:03d}_{id:04d}{ext}'))
                summary.append((index, os.path.join(row[self.params.subject_col_name], row[self.params.study_col_name])))
            else:
                print(f'{row[self.params.subject_col_name]} - missing modalities')

        df = pd.DataFrame(summary, columns=[self.nnunet.subject_column, self.nnunet.original_column])
        df.to_csv(os.path.join(out_scans_path, 'summary.csv'))


if __name__ == '__main__':
    NnunetBptUtils().ichilov_to_nnunet_format(r'D:\users\zeevh\VOL\FOR_SHACHAR\BET_SCANS\summary.csv',
                                              r'D:\users\zeevh\VOL\nnU-Net_data\renamed_data\TASMC\t1_aligned\original')
