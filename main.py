from sre_constants import SUCCESS
from flask import Flask, request
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from flask import jsonify 

async def Predic_model_1(img):
    model =  load_model('meat_xception.h5')
    print("**** load end ******")
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data =   np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    # Replace this with the path to your image
    # image = Image.open('./B.jpg').convert('RGB')
    image =   Image.open(img).convert('RGB')
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size =  (150, 150)
    image =  ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array =  np.asarray(image)
    # Normalize the image
    normalized_image_array =  (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] =  normalized_image_array

    # run the inference
    prediction =  model.predict(data)
    index_1 =  np.argmax(prediction)
    # class_name = class_names[index]
    confidence_score_1= prediction[0][index_1]
    all_confidence_score_1 =  tuple([float(prediction[0][0]),float(prediction[0][1]),float(prediction[0][2])])
    return index_1,confidence_score_1,all_confidence_score_1

async def Predic_model_2(img):
    model =  load_model('meat_cnn.h5')
    print("**** load end ******")
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data =   np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    # Replace this with the path to your image
    # image = Image.open('./B.jpg').convert('RGB')
    image =   Image.open(img).convert('RGB')
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size =  (150, 150)
    image =  ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array =  np.asarray(image)
    # Normalize the image
    normalized_image_array =  (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] =  normalized_image_array

    # run the inference
    prediction =  model.predict(data)
    index_2 =  np.argmax(prediction)
    # class_name = class_names[index]
    confidence_score_2= prediction[0][index_2]
    all_confidence_score_2 =  tuple([float(prediction[0][0]),float(prediction[0][1]),float(prediction[0][2])])
    return index_2,confidence_score_2,all_confidence_score_2

app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
async def hello():
    print("**** get fidle ******")
    img = request.files['file']
    index_1,confidence_score_1,all_confidence_score_1 = await Predic_model_1(img);
    index_2,confidence_score_2,all_confidence_score_2 = await Predic_model_2(img);
    response = { "model_1":{
        "Class":int(index_1),
        "Confidence":float(confidence_score_1),
        "data" : {"class_0_confidence_score":float(all_confidence_score_1[0]),"class_1_confidence_score":float(all_confidence_score_1[1]),"class_2_confidence_score":float(all_confidence_score_1[2]),}
    },
    "model_2":{
        "Class":int(index_2),
        "Confidence":float(confidence_score_2),
        "data" : {"class_0_confidence_score":float(all_confidence_score_2[0]),"class_1_confidence_score":float(all_confidence_score_2[1]),"class_2_confidence_score":float(all_confidence_score_2[2]),}
    }
    }
    return jsonify(response)