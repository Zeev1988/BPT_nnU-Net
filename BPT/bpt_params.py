import os


class BptParams:

    def __init__(self, csv_path=None, out_path=None):
        self.csv_path: str = csv_path
        self.out_path: str = out_path
        self.shrink_output: bool = True

        # Flags specifying whether to save intermediate stage image results
        self.dcm_save_out: bool = False
        self.reg_save_out: bool = False
        self.bet_save_out: bool = True

        self.elastix_exe: str = r"D:\users\Yuval\BET_ZEEV\elastix-5.0.1-win64/elastix.exe"
        self.elastix_params: str = r"D:\users\Yuval\BET_ZEEV\elastix-5.0.1-win64/Parameters_Rigid.txt"
        self.folder_with_parameter_files = os.path.join(os.path.dirname(__file__), "HD_BET",
                                                        "hd-bet_params")
        # Those should match summaries column names
        self.subject_col_name: str = 'Name'
        self.study_col_name: str = 'Study'
        self.is_dcm_col_name: str = 'is_dcm'
        self.t1_name: str = 'T1'
        self.t2_name: str = 'T2'
        self.t1ce_name: str = 'T1C'
        self.flair_name: str = 'FLAIR'
        self.label_name: str = 'Label'

        self.reg_fixed_module: str = self.t1ce_name  # T1, T2, FLAIR - must be valid or registration will fail

        self.modalities = [self.t1_name, self.t1ce_name, self.t2_name, self.flair_name, self.label_name]

        # bet params
        self.n4_fit_level: int = 4
        self.bet_device = 0  # either int (for device id) or 'cpu'
        self.bet_fixed_module: str = self.t1ce_name  # T1C, T2, FLAIR - if not valid, bet will be done separately on each file

        self.perform_reg: bool = True
        self.perform_n4: bool = True
        self.perform_bet: bool = True
        self.perform_resize: bool = False

        self.overwrite: bool = False

    def set_params_from_dict(self, params_external: dict):
        if params_external is None:
            return
        for key, val in params_external.items():
            if key in vars(self):
                setattr(self, key, val)

