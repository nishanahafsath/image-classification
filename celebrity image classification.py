import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Set the path to the directory containing all cropped images
path_to_data = r"C:\Users\user\Downloads\Dataset_Celebrities\cropped"

# Get the list of all image files
image_files = []
for root, dirs, files in os.walk(path_to_data):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(root, file))

# Create a DataFrame with file paths and corresponding labels
df = pd.DataFrame(image_files, columns=['file_path'])
df['label'] = df['file_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

# Determine the number of classes
num_classes = df['label'].nunique()

# Split the data into training, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Use ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up data generators using flow_from_dataframe
target_size = (150, 150)
batch_size = 32

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='file_path', y_col='label', target_size=target_size, batch_size=batch_size, class_mode='categorical')
val_generator = val_datagen.flow_from_dataframe(val_df, x_col='file_path', y_col='label', target_size=target_size, batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(test_df, x_col='file_path', y_col='label', target_size=target_size, batch_size=batch_size, class_mode='categorical')

# Define the CNN Model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  # num_classes is the number of sports persons in your dataset

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Evaluate the model on the test set
eval_result = model.evaluate(test_generator)
print("Test accuracy:", eval_result[1])

# Save the model
model.save("sports_celebrities_model.keras")

# Save the class dictionary
class_dict = {label: i for i, label in enumerate(train_generator.class_indices)}
with open("class_dictionary.json", "w") as f:
    json.dump(class_dict, f)

# Load the class dictionary
with open("class_dictionary.json", "r") as f:
    class_dict = json.load(f)

# Make predictions on a sample image
sample_image_path = r"C:\Users\user\Downloads\Dataset_Celebrities\cropped\virat_kohli\virat_kohli8.png"
sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(150, 150))
sample_image_array = tf.keras.preprocessing.image.img_to_array(sample_image)
sample_image_array = sample_image_array / 255.0  # Normalize the image
sample_image_array = tf.expand_dims(sample_image_array, 0)  # Add batch dimension

# Make predictions
predictions = model.predict(sample_image_array)
predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

# Map the predicted class index to the person's name
predicted_person = [key for key, value in class_dict.items() if value == predicted_class][0]

# Print the predicted person
print("Predicted person:", predicted_person)