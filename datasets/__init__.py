import os
import requests
from zipfile import ZipFile

current_dir = os.getcwd()
dataset_dir = current_dir + '/datasets'
dataset_dir_filelist = os.listdir(dataset_dir)
GTRSB_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'

if 'GTRSB' not in dataset_dir_filelist:
    print("GTRSB folder not found ")
    print('Attempting to download GTSRB_Final_Training_Images.zip from \n', GTRSB_url)
    r = requests.get(GTRSB_url, allow_redirects=True)
    open('GTSRB_Final_Training_Images.zip', 'wb').write(r.content)
    print('Download complete')
    download_file = dataset_dir+'/GTSRB_Final_Training_Images.zip'
    with ZipFile(download_file, 'r') as zip_ref:
        extractall(dataset_dir)
    os.remove(download_file)
