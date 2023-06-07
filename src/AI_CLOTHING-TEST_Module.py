import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('../AI_N1.h5')


# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")                  # Convert to grayscale
    image = image.resize((28, 28))              # Resize to fixed size
    image.show()
    image = np.array(image)                     # Convert to numpy array
    image = image.astype('float32') / 255.0     # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=-1)      # Add channel dimension
    return image


# Load and preprocess the real image
image_path = '../testImages/test-Tshirt.png'
preprocessed_image = preprocess_image(image_path)

# Reshape the preprocessed image to match the input shape expected by the model
input_image = np.expand_dims(preprocessed_image, axis=0)

# Perform prediction
predictions = model.predict(input_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Print the predicted class label
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class label
print('Predicted class label:', predicted_class_label)

