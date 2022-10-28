from sre_constants import SUCCESS
from flask import Flask, request
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from flask import jsonify 

async def Predic(img):
    model =  load_model('keras_model.h5')
    print("**** load end ******")
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data =   np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    # image = Image.open('./B.jpg').convert('RGB')
    image =   Image.open(img).convert('RGB')
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size =  (224, 224)
    image =  ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array =  np.asarray(image)
    # Normalize the image
    normalized_image_array =  (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] =  normalized_image_array

    # run the inference
    prediction =  model.predict(data)
    index =  np.argmax(prediction)
    # class_name = class_names[index]
    confidence_score= prediction[0][index]
    all_confidence_score =  tuple([float(prediction[0][0]),float(prediction[0][1]),float(prediction[0][2])])
    return {index,confidence_score,all_confidence_score}

app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
async def hello():
    print("**** get fidle ******")
    img = request.files['file']
    index,confidence_score,all_confidence_score = await Predic(img);
    response = jsonify(
        Class=int(index),
        Confidence=float(confidence_score),
        data = {"class_0_confidence_score":float(all_confidence_score[0]),"class_1_confidence_score":float(all_confidence_score[1]),"class_2_confidence_score":float(all_confidence_score[2]),}
    )
    return response