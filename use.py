from keras.models import load_model
from keras.utils import load_img, img_to_array

import numpy as np

METTIA = r'qrcods-014_acc-0.789614.h5'
image_to_classification = r"" #your image to classification

model = load_model(METTIA)

picture = load_img(image_to_classification, target_size=(224,224))

picture = img_to_array(picture)

picture = np.expand_dims(picture,axis=0)

picture *=(1./255)

results = model.predict(x=picture,steps=1,verbose=0)

winner_index = np.argmax(results)

print(results.max())

print("Wydaje mi się, że to jest: ",winner_index)
