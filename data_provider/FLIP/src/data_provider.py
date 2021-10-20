# This file will be the main entry point of FLIP dataset container

# The container will provide the logic to interface with FLIP API and manage the datasets accordingly.
from pathlib import Path
import errno
import os
from shutil import copyfile

def dataset_prep(folder_name = Path.cwd() / 'data_provider' / 'FLIP' / 'CROMIS4AD'):
    """
    This function translates the dataset file structure. It creates a new folder with the result with name OLDNAME_READY
    Expected input: one folder per each sample, each one containing another subfolder. This subfolder contains image and label (.nii.gz format)
    Returned output: two folders: images and labels, where the image file and the label file related to the same sample have the same name
    """
    dataset_path = Path(folder_name)
    if not dataset_path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), folder_name)
    
    dataset_ready_path = Path(str(dataset_path) + '_READY')
    dataset_ready_path.mkdir()

    # dataset_ready_path/images and dataset_ready_path/labels
    dataset_ready_images_path = Path(dataset_ready_path / 'images')
    dataset_ready_labels_path = Path(dataset_ready_path / 'labels')
    dataset_ready_images_path.mkdir()
    dataset_ready_labels_path.mkdir()

    sample_dirs = [x for x in dataset_path.glob('*') if x.is_dir()]

    for sample_dir in sample_dirs:
        sample_dir_list = list(sample_dir.glob('*'))
        if len(sample_dir_list) != 1:
            raise ValueError(f"Expected one and only one child element in {sample_dir}, found: {len(sample_dir_list)}")
        
        sample_data_dir = Path(sample_dir_list[0])
        if not sample_data_dir.is_dir():
            raise FileNotFoundError(f"The found child element in {sample_dir} is not a folder")
        
        sample_data_dir_list = list(sample_data_dir.glob('*.nii.gz'))
        if len(sample_data_dir_list) != 2:
            raise ValueError(f"Expected two .nii.gz files in {sample_data_dir}, found: {len(sample_data_dir_list)}")

        if 'label' in sample_data_dir_list[0].name:
            label = sample_data_dir_list[0]
            image = sample_data_dir_list[1]
        elif 'label' in sample_data_dir_list[1].name:
            label = sample_data_dir_list[1]
            image = sample_data_dir_list[0]
        else: 
            raise FileNotFoundError('Expected to found one label file and one image in each data subfolder')
        
        # renaming label file to make it have the same name of the image file
        image_copy = Path(dataset_ready_images_path / image.name)
        label_copy = Path(dataset_ready_labels_path / image.name)

        copyfile(image, image_copy)
        copyfile(label, label_copy)

if __name__ == '__main__':
    dataset_prep()