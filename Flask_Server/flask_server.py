from flask import Flask, render_template, request,url_for
from werkzeug import secure_filename
import os
import keras
import pandas as pd

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import os
import numpy as np
import time

import tensorflow as tf

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

# model = models.load_model('F:/Flask_SIH/keras-retinanet/best_17_28_map.h5')
# model = models.convert_model(model)

labels_to_names = pd.read_csv('F:/Flask_SIH/keras-retinanet/classes.csv',header=None).T.loc[0].to_dict()

OUTPUT = 'F:/Flask_SIH/keras-retinanet/static/output/'
def img_inference(img_path,model):
   image = read_image_bgr(img_path)

  # copy to draw on
   draw = image.copy()
   draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  # preprocess image for network
   image = preprocess_image(image)
   image, scale = resize_image(image)

  # process image
   start = time.time()
   boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
   print("processing time: ", time.time() - start)

  # correct for image scale
   boxes /= scale

  # visualize detections
   for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < 0.4:
         break

      color = label_color(label)

      b = box.astype(int)
      draw_box(draw, b, color=color)

      caption = "{} {:.3f}".format(labels_to_names[label], score)
      draw_caption(draw, b, caption)


   cv2.imwrite(OUTPUT+'draw.png',draw)
#   plt.figure(figsize=(10, 10))
#   plt.axis('off')
#   plt.imshow(draw)
#   plt.show()

app = Flask(__name__)

@app.route('/')
def home():
   if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
      return redirect(url_for('upload_file_yo'))

   return render_template('home.html')

@app.route('/upload')
def upload_file_yo():
   return render_template('flask_template.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      model = models.load_model('F:/Flask_SIH/keras-retinanet/resnet50_csv_06.h5')
      model = models.convert_model(model)
      f = request.files['file']
      f.save('F:/Flask_SIH/keras-retinanet/imgs/'+secure_filename(f.filename))

      x = os.listdir('F:/Flask_SIH/keras-retinanet/imgs')
      img_inference('F:/Flask_SIH/keras-retinanet/imgs/'+x[0],model)

      return 'file saved successfully'
   
@app.route('/output')
def show():
   # full_name = os.listdir(OUTPUT)[0]
   return render_template('show_op.html')

		
if __name__ == '__main__':
   app.run(debug = True)