# Importing libraries
import pix
from pixellib.instance import instance_segmentation

segment_image = instance_segmentation()

# Loading pre-trained model
segment_image.load_model("mask_rcnn_coco.h5")

# Insert any image file that you wnat to run on pre-trained mask-rcnn model and output would be saved in output1.jpg 
segment_image.segmentImage("newyork.jpg", show_bboxes = True, output_image_name = "output1.jpg")
