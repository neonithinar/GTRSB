from data_preprocessing import prepare_directory
import tensorflow as tf
from tensorflow import keras
from configurations.configs import Configurations
import os

Configs = Configurations()
if tf.__version__ == '2.3.0':
    AUTOTUNE = tf.data.experimental.AUTOTUNE
elif tf.__version__ == '2.4.1':
    AUTOTUNE = tf.data.AUTOTUNE

def Create_batch_ds(ds_path, large_ds = False):
    """
    Returns tf.BatchDataset with optimized tf input data pipeline from given dir path
    args:
        ds_path: path to dir
        large_ds (Boolean): if True, disables cache(), else enables caching of ds after each epoch
    """
    ds = keras.preprocessing.image_dataset_from_directory(ds_path,
                                                          seed=42, image_size=(Configs.img_height,
                                                                               Configs.img_width),
                                                          batch_size=Configs.batch_size)
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
    if os.path.exists(Configs.test_dir):
        print("Dataset directory found, creating datasets")
        train_ds = Create_batch_ds(Configs.train_dir)
        val_ds = Create_batch_ds(Configs.val_dir)
        test_ds = Create_batch_ds(Configs.test_dir)

    else:
        print("Dataset directory NOT found, creating dataset dirs")
        prepare_directory.Rename_dirs(Configs.data_dir)
        prepare_directory.Convert_files(Configs.data_dir, Configs.annotations_dir)
        train_path, val_path, test_path = prepare_directory.Create_data_dirs(Configs.data_dir, Configs.output_dir)
        print("creating datasets")
        train_ds = Create_batch_ds(train_path)
        val_ds = Create_batch_ds(val_path)
        test_ds = Create_batch_ds(test_path)

    return train_ds, val_ds, test_ds

