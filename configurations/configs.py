# import json
#
# with open("configs.json") as json_data_file:
#     data = json.load(json_data_file)
# print(data)

img_width = 48
img_height = 48
batch_size = 32
data_dir = "datasets/GTSRB/Final_Training/Images"
annotations_dir = "datasets/GTSRB/annotations"
output_dir = "Datasets/GTRSB/GTRSB_final"
train_dir = "datasets/GTSRB/GTSRB_final/train"
val_dir = "datasets/GTSRB/GTSRB_final/val"
test_dir = "datasets/GTSRB/GTSRB_final/test"
split_ratio = (0.8, 0.1, 0.1)
model_path = "models/CNN_SE_model.h5"
test_image_path = "images/test_img_1.jpg"
ext = [".ppm"]
IMAGES_PATH = "images/"
metrics = "accuracy"
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
CLASS_NAMES = ["00000_speed_limit_20kmph", "00001_speed_limit_30_kmph", "00002_speed_limit_50_kmph",
               "00003_speed_limit_60_kmph", "00004_speed_limit_70_kmph", "00005_speed_limit_80_kmph",
               "00006_end_of_speed_limit", "00007_speed_limit_100_kmph", "00008_speed_limit_120_kmph",
               "00009_no_passing", "00010_no_passing_for_vehicles", "00011_right_of_way_at_the_intersection",
               "00012_priority_road", "00013_yeild", "00014_stop", "00015_no_vehicles", "00016_vehicles_over_34_metres",
               "00017_no_entry", "00018_general_caution", "00019_dangerous_curve_to_left",
               "00020_dangerous_curve_to_right",
               "00021_double_curve", "00022_bumpy_road", "00023_slippery_road", "00024_roads_narrows_on_the_right",
               "00025_road_work", "00026_traffic_signals", "00027_pedestrians", "00028_children_crossing",
               "00029_bicycle_crossing", "00030_beware_of_ice_or_snow", "00031_wild_animals_crossing",
               "00032_end_of_all_speed_and_passing", "00033_turn_right_ahead", "00034_turn_left_ahead",
               "00035_ahead_only", "00036_go_straight_or_right", "00037_go_straight_or_left",
               "00038_keep_right", "00039_keep_left", "00040_roundabout_mandatory", "00041_end_of_passing",
               "00042_end_of_no_passing_by"]


class Configurations():
    """
    Contains basic configurations for the whole project
    """
    def __init__(self):
        self.img_width = 48
        self.img_height = 48
        self.batch_size = 32
        self.data_dir = "datasets/GTSRB/Final_Training/Images"
        self.annotations_dir = "datasets/GTSRB/annotations"
        self.output_dir = "Datasets/GTRSB/GTRSB_final"
        self.train_dir = "datasets/GTSRB/GTSRB_final/train"
        self.val_dir = "datasets/GTSRB/GTSRB_final/val"
        self.test_dir = "datasets/GTSRB/GTSRB_final/test"
        self.split_ratio = (0.8, 0.1, 0.1)
        self.model_path = "models/CNN_SE_model.h5"
        self.test_image_path = "images/test_img_1.jpg"
        self.ext = [".ppm"]
        self.IMAGES_PATH = "images/"
        self.metrics = "accuracy"
        self.name_dict = {"00000": "00_speed_limit_20kmph",
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
        self.CLASS_NAMES = ["00000_speed_limit_20kmph", "00001_speed_limit_30_kmph", "00002_speed_limit_50_kmph",
                       "00003_speed_limit_60_kmph", "00004_speed_limit_70_kmph", "00005_speed_limit_80_kmph",
                       "00006_end_of_speed_limit", "00007_speed_limit_100_kmph", "00008_speed_limit_120_kmph",
                       "00009_no_passing", "00010_no_passing_for_vehicles", "00011_right_of_way_at_the_intersection",
                       "00012_priority_road", "00013_yeild", "00014_stop", "00015_no_vehicles",
                       "00016_vehicles_over_34_metres",
                       "00017_no_entry", "00018_general_caution", "00019_dangerous_curve_to_left",
                       "00020_dangerous_curve_to_right",
                       "00021_double_curve", "00022_bumpy_road", "00023_slippery_road", "00024_roads_narrows_on_the_right",
                       "00025_road_work", "00026_traffic_signals", "00027_pedestrians", "00028_children_crossing",
                       "00029_bicycle_crossing", "00030_beware_of_ice_or_snow", "00031_wild_animals_crossing",
                       "00032_end_of_all_speed_and_passing", "00033_turn_right_ahead", "00034_turn_left_ahead",
                       "00035_ahead_only", "00036_go_straight_or_right", "00037_go_straight_or_left",
                       "00038_keep_right", "00039_keep_left", "00040_roundabout_mandatory", "00041_end_of_passing",
                       "00042_end_of_no_passing_by"]
    def IMAGE_HEIGHT(self):
        return self.image_height




# configs = Configs()
# print(configs.model_path)
