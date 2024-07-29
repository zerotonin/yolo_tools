from pathlib import Path
import cv2
from yolo_tools.training_tools.YoloWrapper import YoloWrapper
import os


# input variables
dataset_path = Path('/home/geuba03p/output/dataset')  # where the YOLO dataset will be
large_field_images_path = Path('/home/geuba03p/output/original_images')  # where the original images
labels_path = Path('/home/geuba03p/output/annotations')  # where the labels are
class_names = ['arena','fly'] # in class order (training data)
config_path = '/home/geuba03p/output/fly_arena.yaml' # create YOLO configuration file
tracker_name = 'fly_arena_3Clin'
epoch_no     = 200


# create the dataset in the format of YOLO
YoloWrapper.create_dataset(large_field_images_path, labels_path, dataset_path)
YoloWrapper.create_config_file(dataset_path, class_names, config_path)


#set the memoryto be expandable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# create pretrained YOLO model and train it using transfer learning
model = YoloWrapper('small')
model.train(config_path, epochs=epoch_no, name=tracker_name)

# make predictions on the validation set
data_to_predict_path = dataset_path/'images'/'val'
val_image_list = list(data_to_predict_path.glob('*.png'))

# save the prediction in a csv file where the bounding boxes should have minimum size
model.predict_and_save_to_csv(val_image_list, path_to_save_csv=f'{tracker_name}.csv', minimum_size=100, threshold=0.25,
                              only_most_conf=True)
# draw bounding boxes from csv
for image in val_image_list:
    model.draw_bbox_from_csv(image, f'{tracker_name}.csv', image.stem)
