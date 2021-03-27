from data_preprocessing import prepare_directory
import tensorflow as tf
from tensorflow import keras
import os


img_width = 48
img_height = 48
batch_size = 32
data_dir = os.path.join("datasets/GTSRB/Final_Training/Images")
annotations_dir = os.path.join('datasets/GTSRB/annotations')
output_dir = 'datasets/GTSRB/GTSRB_final'


train_dir = 'datasets/GTSRB/GTSRB_final/train'
val_dir = 'datasets/GTSRB/GTSRB_final/val'
test_dir = 'datasets/GTSRB/GTSRB_final/test'
AUTOTUNE = tf.data.experimental.AUTOTUNE

def Create_batch_ds(ds_path, large_ds = False):
    """
    Returns tf.BatchDataset with optimized tf input data pipeline from given dir path
    args:
        ds_path: path to dir
        large_ds (Boolean): if True, disables cache(), else enables caching of ds after each epoch
    """
    ds = keras.preprocessing.image_dataset_from_directory(ds_path, seed=42, image_size=(img_height, img_width),
                                                          batch_size=batch_size)
    if large_ds:
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    else:
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds



def Get_datasets():
    """
    returns train_ds, val_ds & test_ds (tensorflow BatchDatasets
    """
    if os.path.exists(test_dir):
        print("Dataset directory found, creating datasets")
        train_ds = Create_batch_ds(train_dir)
        val_ds = Create_batch_ds(val_dir)
        test_ds = Create_batch_ds(test_dir)

    else:
        print("Dataset directory NOT found, creating dataset dirs")
        prepare_directory.Rename_dirs(data_dir)
        prepare_directory.Convert_files(data_dir, annotations_dir)
        train_path, val_path, test_path = prepare_directory.Create_data_dirs(data_dir, output_dir,
                                                                             split_ratio= (0.8, 0.1, 0.1))
        print("creating datasets")
        train_ds = Create_batch_ds(train_path)
        val_ds = Create_batch_ds(val_path)
        test_ds = Create_batch_ds(test_path)

    return train_ds, val_ds, test_ds

