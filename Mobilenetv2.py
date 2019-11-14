#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
from keras.preprocessing import image
import numpy as np
import pandas as pd
import keras
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Model
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


batch_size_train = 32
batch_size_val = 32
batch_size_test = 32
num_classes= 20
STANDARD_SIZE=(224,224)
classes_required = ['Ampalaya', 'Babycorn', 'Bell Pepper', 'Broccoli', 'Cabbage', 'Carrot', 'Eggplant', 'Garlic', 'Ginger', 'Lettuce', 'Mungobean', 'Okra', 'Onion', 'Pechay and Mustard', 'Pole Sitao', 'Snapbeans', 'Soybeans', 'Squash', 'Tomato', 'Upo and Patola']


# In[3]:


from keras.applications import mobilenetv2
mobilenetv2_model = mobilenetv2.MobileNetV2(weights='imagenet')

mobilenetv2_model.summary()


# In[4]:


x = Dense(20, activation='softmax')(mobilenetv2_model.layers[-2].output)

mobilenetv2_model = Model(input=mobilenetv2_model.input, output=x)


# In[5]:


train_path = 'data/Data/Train/'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=STANDARD_SIZE, classes=classes_required, batch_size=batch_size_train)


# In[6]:


val_path = 'data/Data/Test/'
val_batches = ImageDataGenerator().flow_from_directory(val_path, target_size=STANDARD_SIZE, classes=classes_required, batch_size=batch_size_val)


# In[ ]:


import datetime
# batch_size = 32

# initiate RMSprop optimizer
opt1 = keras.optimizers.adam(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
mobilenetv2_model.compile(loss='categorical_crossentropy',
              optimizer=opt1,
              metrics=['accuracy'])

mobilenetv2_plot = mobilenetv2_model.fit_generator(train_batches,
                      steps_per_epoch=200,
                      epochs=50,
                      validation_data=val_batches, 
                      validation_steps=100)
        


# In[15]:


loss, accuracy = mobilenetv2_model.evaluate_generator(generator=val_batches,steps=5)
print('Test Loss: %0.5f Accuracy: %0.5f' % (loss, accuracy))


# In[ ]:


mobilenetv2_model.save_weights("Mobilenetv2model-copy.h5")
print("Saved model to disk")


# In[7]:


mobilenetv2_model.load_weights("Mobilenetv2model.h5")


# In[19]:


get_ipython().system('pip install pydot')


# In[6]:


import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pyd

#Visualize Model

def visualize_model(mobilenetv2_model):
  return SVG(model_to_dot(mobilenetv2_model).create(prog='dot', format='svg'))
#create your model
mobilenetv2_model.load_weights("Mobilenetv2model-copy.h5")
#then call the function on your model
visualize_model(mobilenetv2_model)


# In[8]:


import os
Ampalaya = os.listdir('Data/Test/Ampalaya')
Babycorn = os.listdir('Data/Test/Babycorn')
Bell_Pepper = os.listdir('Data/Test/Bell Pepper')
Broccoli = os.listdir('Data/Test/Broccoli')
Cabbage = os.listdir('Data/Test/Cabbage')
Carrot = os.listdir('Data/Test/Carrot')
Eggplant = os.listdir('Data/Test/Eggplant')
Garlic = os.listdir('Data/Test/Garlic')
Ginger = os.listdir('Data/Test/Ginger')
Lettuce = os.listdir('Data/Test/Lettuce')
Mungobean = os.listdir('Data/Test/Mungobean')
Okra = os.listdir('Data/Test/Okra')
Onion = os.listdir('Data/Test/Onion')
Pechay_Mustard = os.listdir('Data/Test/Pechay and Mustard')
Pole_Sitao = os.listdir('Data/Test/Pole Sitao')
Snapbeans = os.listdir('Data/Test/Snapbeans')
Squash = os.listdir('Data/Test/Squash')
Tomato = os.listdir('Data/Test/Tomato')
Upo_Patola = os.listdir('Data/Test/Upo and Patola')
Soybeans = os.listdir('Data/Test/Soybeans')


# In[10]:


path = 'Data/Test'
class_predict = 'Ampalaya'
img_width, img_height = 224,224
error = 0
for i in range(5,10):
    img = image.load_img(path + "/" + class_predict + "/" + str(Ampalaya[i]), target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    Y_pred = mobilenetv2_model.predict(img)
    y_pred = np.argmax(Y_pred, axis=1)
    if y_pred[0] != 1:
        error += 1
    print("Predicted: " + str(classes_required[y_pred[0]]) + "  Actual:" + class_predict)
print(error)


# In[11]:


img_orig = image.load_img("Data/Validation/Ampalaya.jpg", target_size = (img_width, img_height))
img = image.img_to_array(img_orig)
img = np.expand_dims(img, axis = 0)

Y_pred = mobilenetv2_model.predict(img)
y_pred = np.argmax(Y_pred, axis=1)
y_pred
labels = ['Ampalaya', 'Babycorn', 'Bell Pepper', 'Broccoli', 'Cabbage', 'Carrot', 'Eggplant', 'Garlic', 'Ginger', 'Lettuce', 'Mungobean', 'Okra', 'Onion', 'Pechay and Mustard', 'Pole Sitao', 'Snapbeans', 'Soybeans', 'Squash', 'Tomato', 'Upo and Patola']


# In[12]:


image_label = labels[y_pred[0]]


# In[13]:


plt.title(image_label)
plt.imshow(img_orig)
plt.show()







