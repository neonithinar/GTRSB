import prepare_directory
import tensorflow as tf
from tensorflow import keras
import os


img_width = 48
img_height = 48
batch_size = 32
train_dir = 'datasets/GTRSB/GTRSB_final/train'
val_dir = 'datasets/GTRSB/GTRSB_final/val'
test_dir = 'datasets/GTRSB/GTRSB_final/test'
AUTOTUNE = tf.data.AUTOTUNE

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
        train_ds = Create_batch_ds(train_dir)
        val_ds = Create_batch_ds(val_dir)
        test_ds = Create_batch_ds(test_dir)

    else:
        prepare_directory.Rename_dirs(data_dir)
        prepare_directory.Convert_files(data_dir, annotations_dir)
        train_path, val_path, test_path = prepare_directory.Create_data_dirs(data_dir, ouput_dir,
                                                                             split_ratio= (0.8, 0.1, 0.1))

        train_ds = Create_batch_ds(train_path)
        val_ds = Create_batch_ds(val_path)
        test_ds = Create_batch_ds(test_path)

    return train_ds, val_ds, test_ds

