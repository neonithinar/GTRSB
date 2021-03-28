import os
import splitfolders
import cv2
from configurations.configs import Configurations

Configs = Configurations()

def Rename_dirs(data_dir):
    """
    Renames the folders in datasets/GTSRB/Final_Training/Images
    from abstract numbered classes to actual traffic sign names
    (eg: '00034' to 34_turn_left_ahead)

    args: data_dir: path to directory containing the sub-folders to classes

    """

    for root, dirs, files in os.walk(data_dir, topdown=False):
        dirs = sorted(dirs)
        for dir_name in dirs:
            if dir_name not in Configs.name_dict.values():
                # print(root +"/"+ name) # for future logging
                # print(name_dict[str(name)])
                rename_dir = root + "/" + dir_name
                final_name = root + "/" + Configs.name_dict[str(dir_name)]
                os.rename(rename_dir, final_name)
            else:
                print('folder already exists')

    return print("Folders Renamed")

def Convert_files(data_dir, annotations_dir):
    """
    Renames all the files with a 'prefix_' + filename for
    tf.keras.preprocessing.image_dataset_from_directory() batch dataset generator
    Converts all the .ppm image files in data_dir from '.ppm' to '.jpg'
    moves every other file to annotations folder

    args: data_dir : path to image directory.
          annotation_dir : path to move non image files

    """

    for root, dirs, files in os.walk(data_dir, topdown=False):
        dirs = sorted(dirs)
        for dir_name in dirs:

            for dir_root, _, sub_dir_files in os.walk(os.path.join(root + "/" + dir_name)):
                sub_dir_files = sorted(sub_dir_files)
                for filename in sub_dir_files:
                    if 'prefix' not in filename.split('_'):
                        rename_file = dir_root + "/" + filename
                        new_name = dir_root + "/" + "prefix_" + filename
                        _, ext = os.path.splitext(filename)
                        if ext == ".ppm":
                            os.rename(rename_file, new_name)
                            i = cv2.imread(new_name)
                            cv2.imwrite((new_name.strip(".ppm") + ".jpg"), i)
                            print("files converted")
                        else:
                            os.rename(rename_file, new_name)
                            new_file = annotations_dir + '/' + 'prefix_' + filename
                            os.replace(new_name, new_file)
                            # os.remove(rename_file)


    return print("All files have been converted, Renamed and moved succesfully")


def Create_data_dirs(data_dir, output_dir, split_ratio= Configs.split_ratio):
    """
    Creates train_dir, val_dir, test_dir folders with shuffled data from data_dir
    according to split_ratio arg

    args:
        data_dir: source directory containing image file data
        ouput_dir: destination dir. if the path doesn't exist, it will be created automatically
        split_ratio: train:val:test split ratio

    returns:
        tuple (train_path, validation_path, test_path)
    """

    if os.path.exists(output_dir):
        os.makedirs(output_dir + '/train')
        os.makedirs(output_dir + '/test')
        os.makedirs(output_dir + '/val')
        splitfolders.ratio(data_dir, output=output_folder, seed=42, ratio=split_ratio)

    else:
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/train')
        os.makedirs(output_dir + '/test')
        os.makedirs(output_dir + '/val')
        splitfolders.ratio(data_dir, output=output_dir, seed=42, ratio=split_ratio)

    train_dir_path = os.path.join(output_dir + '/train')
    val_dir_path = os.path.join(output_dir + '/val')
    test_dir_path = os.path.join(output_dir + '/test')

    return train_dir_path, val_dir_path, test_dir_path

