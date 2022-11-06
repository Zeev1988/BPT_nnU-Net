import pandas as pd
import os
import shutil
from BPT.bpt_params import BptParams


class BptUtils:
    def __init__(self, params=None):
        self.params = params if params else BptParams()

    def df2data(self, df):
        data = {}
        for index, row in df.iterrows():
            modules = {self.params.t1_name: row[self.params.t1_name],
                       self.params.t1ce_name: row[self.params.t1ce_name],
                       self.params.t2_name: row[self.params.t2_name],
                       self.params.flair_name: row[self.params.flair_name],
                       self.params.label_name: row[self.params.label_name]}
            if row[self.params.subject_col_name] not in data:
                data[row[self.params.subject_col_name]] = []
            data[row[self.params.subject_col_name]].append({row[self.params.study_col_name]: modules})
        return data

    def generate_data(self, subject: tuple):
        study = subject[1]
        values = list(study.values())[0]
        return {self.params.subject_col_name: subject[0],
                self.params.study_col_name: list(study.keys())[0],
                self.params.t1_name: values[self.params.t1_name],
                self.params.t1ce_name: values[self.params.t1ce_name],
                self.params.t2_name: values[self.params.t2_name],
                self.params.flair_name: values[self.params.flair_name],
                self.params.label_name: values[self.params.label_name]}

    def data2csv(self, data: list, output_dir: str):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)

    def rename_copy_files(self, subject: tuple, output_dir: str, modalities: list):
        study = subject[1]
        modules = list(study.values())[0]
        for modality in modalities:
            if modules[modality]:
                in_dir = os.path.dirname(modules[modality])
                ext = '.nii.gz' if modules[modality].endswith('.gz') else '.nii'

                if in_dir != output_dir:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    shutil.copy(modules[modality], output_dir)

                new_file_name = os.path.join(output_dir, f'{modality}{ext}')
                os.rename(os.path.join(output_dir, os.path.basename(modules[modality])), new_file_name)
                modules[modality] = new_file_name


if __name__ == '__main__':
    pass
