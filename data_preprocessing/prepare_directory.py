import os
import splitfolders
import cv2

name_dict = {"00000": "00_speed_limit_20kmph",
             "00001": "01_speed_limit_30_kmph",
             "00002": "02_speed_limit_50_kmph",
             "00003": "03_speed_limit_60_kmph",
             "00004": "04_speed_limit_70_kmph",
             "00005": "05_speed_limit_80_kmph",
             "00006": "06_end_of_speed_limit",
             "00007": "07_speed_limit_100_kmph",
             "00008": "08_speed_limit_120_kmph",
             "00009": "09_no_passing",
             "00010": "10_no_passing_for_vehicles",
             "00011": "11_right_of_way_at_the_intersection",
             "00012": "12_priority_road",
             "00013": "13_yeild",
             "00014": "14_stop",
             "00015": "15_no_vehicles",
             "00016": "16_vehicles_over_34_metres",
             "00017": "17_no_entry",
             "00018": "18_general_caution",
             "00019": "19_dangerous_curve_to_left",
             "00020": "20_dangerous_curve_to_right",
             "00021": "21_double_curve",
             "00022": "22_bumpy_road",
             "00023": "23_slippery_road",
             "00024": "24_roads_narrows_on_the_right",
             "00025": "25_road_work",
             "00026": "26_traffic_signals",
             "00027": "27_pedestrians",
             "00028": "28_children_crossing",
             "00029": "29_bicycle_crossing",
             "00030": "30_beware_of_ice_or_snow",
             "00031": "31_wild_animals_crossing",
             "00032": "32_end_of_all_speed_and_passing",
             "00033": "33_turn_right_ahead",
             "00034": "34_turn_left_ahead",
             "00035": "35_ahead_only",
             "00036": "36_go_straight_or_right",
             "00037": "37_go_straight_or_left",
             "00038": "38_keep_right",
             "00039": "39_keep_left",
             "00040": "40_roundabout_mandatory",
             "00041": "41_end_of_passing",
             "00042": "42_end_of_no_passing_by"}


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
            if dir_name not in name_dict.values():
                # print(root +"/"+ name) # for future logging
                # print(name_dict[str(name)])
                rename_dir = root + "/" + dir_name
                final_name = root + "/" + name_dict[str(dir_name)]
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


def Create_data_dirs(data_dir, output_dir, split_ratio= (0.8, 0.1, 0.1)):
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
        splitfolders.ratio(data_dir, output=output_folder, seed=42, ratio=(0.8, 0.1, 0.1))

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

