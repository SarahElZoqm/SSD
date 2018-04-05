import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession (config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
#by3ml whetining lel image b2no y4el mn 2l sora means w resize leha w byzbt 7ga fl bbox 
image_4d = tf.expand_dims(image_pre, 0)#[1, height, width, channels]

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)
# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=6, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
# Test on some demo image and visualize output.
##input is images in demo file and output is images in images directory and output video (called (output.avi) in notebooks directory) if you remove comment
path = '../demo/'
image_names = sorted(os.listdir(path))
##############if you want output is video remove(#) from 2 line.########################
#fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
#out= cv2.VideoWriter('output.avi',fourcc,10,(1280,720)) 
########################################################################################
for i in range(8):
 
    img = mpimg.imread(path + image_names[i])
    
    rclasses, rscores, rbboxes =  process_image(img)
    
       
    #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    img=visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    ##this do to save images in images directory.
    ##note that you should create images directory in notebooks directory to save images in it.
    imgName="images/"+str(i)+".jpg"
    cv2.imwrite(imgName,img)
##############if you want output is video remove(#) from 3 line.########################
    #img= cv2.resize(img, (1280,720))
    #out.write(img)
#out.release()
#########################################################################################

##input is video and output is images in images directory and output video called (output.avi) in notebooks directory
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out= cv2.VideoWriter('output.avi',fourcc,25,(1280,720))
##you should put your video in notebooks directory
##(My_Movie.mp4) this name of video.if video in another notebooks directory ,you must write path of video 
strm = cv2.VideoCapture("My_Movie.mp4")  
i=0
while (True):
   ret,img=strm.read()
   if(ret==False):
       break
   rclasses, rscores, rbboxes =  process_image(img)
   img=visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
   ##this do to save images in images directory.
   ##note that you should create images directory in notebooks directory to save images in it.
   imgName="images/"+str(i)+".jpg"
   cv2.imwrite(imgName,img)
   i=i+1
   img= cv2.resize(img, (1280,720))
   out.write(img)
   #key = cv2.waitKey(1) & 0xFF
   #if key == ord("q"):
        #break
        
out.release()

