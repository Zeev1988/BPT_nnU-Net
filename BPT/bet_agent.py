import SimpleITK as sitk
import os
from BPT.HD_BET.run import run_hd_bet
import logging
import re


def n4_bias_correction(m_path, out_path):
    inputImage = sitk.ReadImage(m_path)
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # numberFilltingLevels = n4_fit_lvl
    # corrector.SetMaximumNumberOfIterations([int(sys.argv[5])] * numberFittingLevels)
    output = corrector.Execute(inputImage, maskImage)
    sitk.WriteImage(output, out_path)


def bet(subject, output_dir, fixed_module, do_n4, shrink, modalities, label):
    ext = ".nii.gz" if shrink else '.nii'
    mask_image = None
    mask_path = None
    study = subject[1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modules = list(study.values())[0]
    fixed_exists = os.path.exists(modules[fixed_module]) and (modules[fixed_module].endswith('.nii') or
                                                              modules[fixed_module].endswith('.nii.gz'))

    if not fixed_exists:
        logging.error(f"BET - fixed bet file is not valid (not found or not nifty) ")
        print(f"BET - Error: fixed bet file was not found (not found or not nifty) {modules[fixed_module]}")
        logging.error("BET - will continue without a fixed module!!!")
        print("BET - Error: will continue without a fixed module!!!")

    # move fixed module to beginning of list so it will be the first to be processed
    # modality_list = ['T1', 'T1C', 'T2', 'FLAIR']
    modality_list = modalities.copy()
    modality_list.insert(0, modality_list.pop(modality_list.index(fixed_module)))

    for modality in modality_list:
        if modules[modality]:
            if not os.path.exists(modules[modality]):
                logging.error(f"BET - file was not found {modules[modality]}")
                print(f"BET - Error: file was not found {modules[modality]}")
                continue

            if not modules[modality].endswith('.nii') and not modules[modality].endswith('.nii.gz'):
                logging.error(f"BET - file is not of type nifty {modules[modality]}")
                print(f"BET - Error: file is not of type nifty {modules[modality]}")
                continue

            out_path = os.path.join(output_dir, os.path.basename(modules[modality]))
            out_path = re.sub(r"(.nii.gz|.nii)+", ext, out_path)
            in_path = modules[modality]
            if do_n4 and modality != label:
                print(f'N4 bias field correction started for modality {modality}\n')
                logging.info(f"N4 correction started - input scan: {in_path}, output scan: {out_path}")
                n4_bias_correction(in_path, out_path)
                in_path = out_path
                # print('N4 bias field correction ended\n')
                logging.info("n4 correction ended")

            if modality == fixed_module or (not fixed_exists and modality != label):
                print(f'Generating mask for brain extraction using modality {modality}\n')
                logging.info(f"Brain extraction mask generation process started using modality {modality}")
                run_hd_bet(in_path, out_path, keep_mask=True)
                mask_path = out_path.replace(".nii", "_mask.nii")
                mask_image = sitk.ReadImage(mask_path)
            else:
                image = sitk.ReadImage(in_path)
                if modality != label:
                    try:
                        image = sitk.Mask(image, sitk.Cast(mask_image, sitk.sitkInt8), maskingValue=0, outsideValue=0)
                    except RuntimeError:
                        # Increase the tolerance in case the execution fails
                        sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(1e-5)
                        sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(1e-5)
                        try:
                            image = sitk.Mask(image, sitk.Cast(mask_image, sitk.sitkInt8), maskingValue=0, outsideValue=0)
                        except RuntimeError:
                            # If the execution fails after increasing the tolerance, use the direction from the mask image
                            image.SetDirection(mask_image.GetDirection())
                            image = sitk.Mask(image, sitk.Cast(mask_image, sitk.sitkInt8), maskingValue=0, outsideValue=0)
                sitk.WriteImage(image, out_path)

    os.remove(mask_path)

    logging.info("BET - brain extraction tool ended")
