  
from flask import Flask, render_template, request,url_for, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import sys
import zipfile
import PIL
from matplotlib import pyplot as plt
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as matplot
import cv2
import numpy as np
import label_map_util
import visualization_utils as vis_util

PATH_TO_CKPT = 'G:/PCI-Detector/Flask_Server/inception.pb'
PATH_TO_LABELS = 'G:/PCI-Detector/Flask_Server/label_map.pbtxt'


NUM_CLASSES = 8

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,                                                                use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)



        

# def box_util(x):
#    bbox =[]
#    cnt = 0
#    for i in range(len(x)):
#       if i ==0 :
#          pass
#       if i>0:
#          cnt += 1
#          if cnt <= 4:
#             bbox.append(x[i])
#          else:
#             cnt = 0
#    return bbox

# def calculate_area(bbox):
#    TOTAL= 600*600
#    size = 4 
#    c = 0
#    total_damaged_area = 0
#    while True:
#       c += 1
#       if c == len(bbox)/4 and len(bbox) == 4:
#          pass
#       elif c>= len(bbox)/4:
#          break
#       if len(bbox) == 4:
#          box = bbox
#       else:
#          box = bbox[c*size:(c+1)*size]
      
#       area = (int(box[2])-int(box[0]))*(int(box[3])-int(box[1]))
#       total_damaged_area += area
#       if len(bbox)> 0:
#          return ((total_damaged_area*100)/TOTAL),len(bbox)
#       else:
#          return (0,0)


# def calculate_pci(bbox):
#    bbox = box_util(bbox)
#    try:
#       p,n = calculate_area(bbox)
#    except:
#       p,n = 0,0

#    if (n > 3 and p > 10.0):
#       pci = 5
#    elif (10.0>p>7.5):
#       pci = 4
#    elif (7.5>p>6.0):
#       pci = 3
#    elif (6.0>p>4.5):
#       pci = 2
#    elif (4.5>p>3.0):
#       pci = 1
#    else:
#       pci = 0
   
#    return pci

def calculate_pci(scores, classes):
    score = scores[scores > 0.1]
    clas = classes[scores > 0.1]
#         print(clas)
    label = [1,1,1,1,16,16,1,1]
    confidence = [14, 27, 40]
    pci = 100
    for s in range(score.size):
        if score[s] <= 0.33:
            pci -= label[int(clas[s]-1)]*confidence[0]*0.04
        elif score[s] > 0.33 and score[s] <=0.66:
            pci -= label[int(clas[s]-1)]*confidence[1]*0.04
        else:
            pci -= label[int(clas[s]-1)]*confidence[2]*0.04
    if pci <= 15:
        pci = 15
    return pci


# labels_to_names = pd.read_csv(PATH+ 'classes.csv',header=None).T.loc[0].to_dict()
PATH = 'G:/PCI-Detector/'


IMAGE_SIZE = (12, 8)
def img_inference(img_path):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = PIL.Image.open(img_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np,axis=0)

            (boxes,scores,classes,num) = sess.run([detection_boxes,detection_scores,detection_classes,num_detections],
            feed_dict={image_tensor: image_np_expanded})

            pci = calculate_pci(scores,classes)
            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    min_score_thresh=0.5,
                    use_normalized_coordinates=True,
                    line_thickness=8)

            plt.figure(figsize=IMAGE_SIZE)
            plt.title('Pavement Condition Index (PCI): ' + str(round(pci)))
            plt.imshow(image_np)
            name = img_path.split('/')[-1].split('.')[0]

            op_name = PATH+'/imgs/'+name+'_OUTPUT.png'
            cv2.imwrite(op_name,image_np)
   # pci = calculate_pci(boxes[0])
    return (op_name, pci)
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
    #   model = models.load_model(PATH+'resnet50_csv_06.h5')
    #   model = models.convert_model(model)
      f = request.files['file']
      f.save(PATH+'imgs/'+secure_filename(f.filename))

      x = os.listdir(PATH+'imgs')
      output_path,pci = img_inference(PATH+'/imgs/'+x[0])
    #   pci = calculate_pci(bbox)
      op = {'FilePath':output_path, 'PCI':pci}
      os.remove(PATH+'imgs/'+x[0])
      return jsonify(op)
   


		
if __name__ == '__main__':
   app.run(debug = True)