from flask import Flask, render_template, jsonify, request,url_for
# from werkzeug import secure_filename
import os
import keras
import pandas as pd
import cloudinary as Cloud
import cloudinary.uploader
import cloudinary.api
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import json

import cv2
import os
import numpy as np
import time
import requests

import tensorflow as tf

upload_preset = 'employeeApp'
cloud_name = 'pathtut'

cloudinary.config( 
  cloud_name = cloud_name, 
  api_key = "986796431438695", 
  api_secret = "39BSSdfbjSfyEA2GrUV_u2ftFY4" 
)

def box_util(x):
   bbox =[]
   cnt = 0
   for i in range(len(x)):
      if i ==0 :
         pass
      if i>0:
         cnt += 1
         if cnt <= 4:
            bbox.append(x[i])
         else:
            cnt = 0
   return bbox

def calculate_area(bbox):
   TOTAL= 600*600
   size = 4 
   c = 0
   total_damaged_area = 0
   while True:
      c += 1
      if c == len(bbox)/4 and len(bbox) == 4:
         pass
      elif c>= len(bbox)/4:
         break
      if len(bbox) == 4:
         box = bbox
      else:
         box = bbox[c*size:(c+1)*size]
      
      area = (int(box[2])-int(box[0]))*(int(box[3])-int(box[1]))
      total_damaged_area += area
      if len(bbox)> 0:
         return ((total_damaged_area*100)/TOTAL),len(bbox)
      else:
         return (0,0)

def calculate_pci(bbox):
   bbox = box_util(bbox)
   try:
      p,n = calculate_area(bbox)
   except:
      p,n = 0,0

   if (n > 3 and p > 10.0):
      pci = 5
   elif (10.0>p>7.5):
      pci = 4
   elif (7.5>p>6.0):
      pci = 3
   elif (6.0>p>4.5):
      pci = 2
   elif (4.5>p>3.0):
      pci = 1
   else:
      pci = 0
   
   return pci

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)
PATH = "/home/harshitdixit/interests/react_native/path/model/PCI-Detector/keras-retinanet"
keras.backend.tensorflow_backend.set_session(get_session())

# model = models.load_model('F:/Flask_SIH/keras-retinanet/best_17_28_map.h5')
# model = models.convert_model(model)

labels_to_names = pd.read_csv('{}/classes.csv'.format(PATH),header=None).T.loc[0].to_dict()

OUTPUT = '{}/static/output/'.format(PATH)
def img_inference(img_path,model):
   pred = []
   image = read_image_bgr(img_path)
   name = img_path.split('/')[-1].split('.')[0]
  # copy to draw on
   draw = image.copy()
   draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  # preprocess image for network
   image = preprocess_image(image)
   # image, scale = resize_image(image)
   image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_AREA)
  # process image
   start = time.time()
   boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
   print("processing time: ", time.time() - start)

  # correct for image scale
   # boxes /= scale

  # visualize detections
   for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < 0.4:
         break

      color = label_color(label)

      b = box.astype(int)
      pred.append(str(label))
      for i in b:
         pred.append(str(i))

      draw_box(draw, b, color=color)

      caption = "{} {:.3f}".format(labels_to_names[label], score)
      draw_caption(draw, b, caption)

   op_name = OUTPUT+name+'_OUTPUT.png'
   cv2.imwrite(op_name,draw)
   # pci = calculate_pci(boxes[0])
   return (op_name, pred)
#   plt.figure(figsize=(10, 10))
#   plt.axis('off')
#   plt.imshow(draw)
#   plt.show()

app = Flask(__name__)

	
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':
      model = models.load_model('{}/resnet50_csv_06.h5'.format(PATH))
      model = models.convert_model(model)
      payload = request.get_json()
      print(payload)
      imgUrl = payload['imageUrl']
      response = requests.get(imgUrl, stream = True)
      if response.status_code == 200:
         file = open("{}/imgs/sample_png.png".format(PATH), "wb")
         file.write(response.content)
         file.close()
      # f = request.files['file']
      # f.save('{}/imgs/'.format(PATH)+(f.filename))
         path_img, bbox = img_inference('{}/imgs/sample_png.png'.format(PATH), model)
         pipeshelf = Cloud.uploader.unsigned_upload(path_img, upload_preset)
         print(pipeshelf['url'])
         pci = calculate_pci(bbox)
         return jsonify({ 'damageImageUrl': pipeshelf['url'],'pci':pci})
      # x = os.listdir('{}/imgs'.format(PATH))
      # img_inference('{}/imgs/'.format(PATH)+x[0],model)

		
if __name__ == '__main__':
   app.run(debug = True)