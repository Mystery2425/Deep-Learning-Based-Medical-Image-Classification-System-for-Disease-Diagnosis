import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#  Download data from Kaggle
!pip install kaggle
from google.colab import files

Requirement already satisfied: kaggle in /usr/local/lib/python3.10/dist-packages (1.6.17)
Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.10/dist-packages (from kaggle) (1.16.0)
Requirement already satisfied: certifi>=2023.7.22 in /usr/local/lib/python3.10/dist-packages (from kaggle) (2024.8.30)
Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.8.2)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.32.3)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from kaggle) (4.66.5)
Requirement already satisfied: python-slugify in /usr/local/lib/python3.10/dist-packages (from kaggle) (8.0.4)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.2.3)
Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from kaggle) (6.1.0)
Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->kaggle) (0.5.1)
Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.10/dist-packages (from python-slugify->kaggle) (1.3)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.10)

import os
import shutil

# Create a .kaggle folder
os.makedirs('~/.kaggle', exist_ok=True)

#Verify if the file exists
if os.path.exists('kaggle.json'):
    # Move the kaggle.json file to the folder
    shutil.move('kaggle.json', '~/.kaggle/kaggle.json')

    # Set the correct permissions.
    os.chmod('~/.kaggle/kaggle.json', 600)
else:
    #Print error message if file not found
    print("Error: kaggle.json not found in the current directory.")

!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

Dataset URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
License(s): other
Downloading chest-xray-pneumonia.zip to /content
100% 2.29G/2.29G [01:22<00:00, 31.1MB/s]
100% 2.29G/2.29G [01:22<00:00, 30.0MB/s]

import zipfile

# Decompress the dataset
with zipfile.ZipFile('chest-xray-pneumonia.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')

#Preparing training and testing data
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    validation_split=0.2)  # Use 20% to verify

train_generator = train_datagen.flow_from_directory(
    'data/chest_xray/train',  # Make sure the data is in this path.
    target_size=(150, 150),  # Resize images
    batch_size=32,
    class_mode='binary',  # binary category
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/chest_xray/train',  # Make sure the data is in this path.
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

Found 4173 images belonging to 2 classes.
Found 1043 images belonging to 2 classes.

#Build the modelmodel
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Use Dropout to reduce overfitting
    Dense(1, activation='sigmoid')  # Binary output
])

# Compile the modelmodel.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

Epoch 1/10

/usr/local/lib/python3.10/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()

131/131 ━━━━━━━━━━━━━━━━━━━━ 294s 2s/step - accuracy: 0.7823 - loss: 0.4856 - val_accuracy: 0.8888 - val_loss: 0.2737
Epoch 2/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 293s 2s/step - accuracy: 0.8856 - loss: 0.2771 - val_accuracy: 0.8773 - val_loss: 0.3802
Epoch 3/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 315s 2s/step - accuracy: 0.9130 - loss: 0.2238 - val_accuracy: 0.8936 - val_loss: 0.2471
Epoch 4/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 316s 2s/step - accuracy: 0.9093 - loss: 0.2179 - val_accuracy: 0.8974 - val_loss: 0.2382
Epoch 5/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 327s 2s/step - accuracy: 0.9220 - loss: 0.1983 - val_accuracy: 0.9089 - val_loss: 0.2271
Epoch 6/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 317s 2s/step - accuracy: 0.9154 - loss: 0.2094 - val_accuracy: 0.9166 - val_loss: 0.2108
Epoch 7/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 339s 2s/step - accuracy: 0.9349 - loss: 0.1769 - val_accuracy: 0.9262 - val_loss: 0.1933
Epoch 8/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 281s 2s/step - accuracy: 0.9350 - loss: 0.1749 - val_accuracy: 0.9319 - val_loss: 0.1844
Epoch 9/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 287s 2s/step - accuracy: 0.9329 - loss: 0.1747 - val_accuracy: 0.9377 - val_loss: 0.1762
Epoch 10/10
131/131 ━━━━━━━━━━━━━━━━━━━━ 283s 2s/step - accuracy: 0.9426 - loss: 0.1502 - val_accuracy: 0.9463 - val_loss: 0.1513

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot training results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Display some images with predictions
import random
sample_images, _ = next(validation_generator)
predictions = model.predict(sample_images)

for i in range(5):
    plt.imshow(sample_images[i])
    plt.title(f'Predicted: {"Pneumonia" if predictions[i] > 0.5 else "Normal"}')
    plt.axis('off')
    plt.show()

# Save the trained model
model.save('chest_xray_model.h5')

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
import zipfile
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from google.colab import files

from google.colab import files
files.upload()

# Make sure the kaggle.json file exists in the current directory.
# If not, download it from your Kaggle account and upload it to the Colab environment.
# Then, create the destination directory if it doesn't exist

import os
os.makedirs('/root/.kaggle/', exist_ok=True) # Create the directory if it doesn't exist

# Now move the file
shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
!chmod 600 /root/.kaggle/kaggle.json
!pip install kaggle

Requirement already satisfied: kaggle in /usr/local/lib/python3.10/dist-packages (1.6.17)
Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.10/dist-packages (from kaggle) (1.16.0)
Requirement already satisfied: certifi>=2023.7.22 in /usr/local/lib/python3.10/dist-packages (from kaggle) (2024.8.30)
Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.8.2)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.32.3)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from kaggle) (4.66.5)
Requirement already satisfied: python-slugify in /usr/local/lib/python3.10/dist-packages (from kaggle) (8.0.4)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.2.3)
Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from kaggle) (6.1.0)
Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->kaggle) (0.5.1)
Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.10/dist-packages (from python-slugify->kaggle) (1.3)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.10)

# Download X-ray dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
with zipfile.ZipFile('chest-xray-pneumonia.zip', 'r') as zip_ref:
    zip_ref.extractall('chest_xray')

Dataset URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
License(s): other
chest-xray-pneumonia.zip: Skipping, found more recently modified local copy (use --force to force download)

!ls chest_xray

chest_xray

# Replace 'chest-xray-pneumonia.zip' with the actual name of your downloaded zip file
with zipfile.ZipFile('chest-xray-pneumonia.zip', 'r') as zip_ref:
    zip_ref.extractall('chest_xray')

#Preparing training and testing data
train_dir = 'chest_xray/chest_xray/train' # Updated path to include the extracted directory
val_dir = 'chest_xray/chest_xray/val' # Updated path to include the extracted directory

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Building a neural network model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

Found 5216 images belonging to 2 classes.
Found 16 images belonging to 2 classes.

/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

from tensorflow.keras.layers import Input

# Building a neural network model
model = Sequential([
    Input(shape=(150, 150, 3)),  # Use Input object as first layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

Epoch 1/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 290s 2s/step - accuracy: 0.7730 - loss: 0.5814 - val_accuracy: 0.6250 - val_loss: 1.2727
Epoch 2/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 283s 2s/step - accuracy: 0.9022 - loss: 0.2439 - val_accuracy: 0.7500 - val_loss: 0.7440
Epoch 3/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 326s 2s/step - accuracy: 0.9207 - loss: 0.1925 - val_accuracy: 0.8750 - val_loss: 0.4452
Epoch 4/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 282s 2s/step - accuracy: 0.9308 - loss: 0.1772 - val_accuracy: 0.7500 - val_loss: 0.6577
Epoch 5/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 282s 2s/step - accuracy: 0.9403 - loss: 0.1662 - val_accuracy: 0.8125 - val_loss: 0.4520
Epoch 6/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 289s 2s/step - accuracy: 0.9445 - loss: 0.1467 - val_accuracy: 0.6875 - val_loss: 0.7643
Epoch 7/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 284s 2s/step - accuracy: 0.9459 - loss: 0.1380 - val_accuracy: 0.8125 - val_loss: 0.6009
Epoch 8/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 288s 2s/step - accuracy: 0.9454 - loss: 0.1538 - val_accuracy: 0.8750 - val_loss: 0.4754
Epoch 9/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 281s 2s/step - accuracy: 0.9555 - loss: 0.1197 - val_accuracy: 0.8125 - val_loss: 0.4533
Epoch 10/10
163/163 ━━━━━━━━━━━━━━━━━━━━ 289s 2s/step - accuracy: 0.9585 - loss: 0.1173 - val_accuracy: 0.8125 - val_loss: 0.4288

# Drawing accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save form
model.save('chest_xray_model.h5')

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

import os
import random
from tensorflow.keras.preprocessing import image

# Select a random image from the verification set.
validation_class_dir = os.path.join(val_dir, 'PNEUMONIA')  # You can change 'PNEUMONIA' to 'NORMAL' to use an image from the normal category.
random_image = random.choice(os.listdir(validation_class_dir))
image_path = os.path.join(validation_class_dir, random_image)

# Function to predict image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

# Predict using selected image
result = predict_image(image_path)
print(f'Prediction for the selected image: {result}')

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 141ms/step
Prediction for the selected image: Pneumonia

Interactive User Interface Using Gradio

!pip install gradio # Install the gradio package
import gradio as gr

def predict_image_gradio(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

interface = gr.Interface(fn=predict_image_gradio, inputs="image", outputs="text")
interface.launch()

Create an automated report that displays some statistics about your X-ray data, such as the number of healthy and pneumonia-infected samples. Using a library like matplotlib or seaborn to create graphs that display the distribution of the data.

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def predict_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction[0][0]

    return class_label, f"Confidence: {confidence:.2f}"

# Gradio interface
iface = gr.Interface(fn=predict_image, inputs="image", outputs=["text", "text"])

# Add an interface to analyze a set of images
def predict_images(imgs):
    results = []
    for img in imgs:
        result = predict_image(img)
        results.append(result)
    return results

multi_image_interface = gr.Interface(fn=predict_images, inputs="image", outputs="label", title="Batch Analysis")

# Launch Gradio interface
iface.launch(share=True)




