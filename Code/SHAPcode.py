import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

data_dir = 'cancerbrain'
data_dir2 = 'brain_testing'
filepaths = []
labels = []

# Categories for classification
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load data from both directories
for data_directory in [data_dir, data_dir2]:
    for category in categories:
        category_path = os.path.join(data_directory, category)
        if os.path.exists(category_path):
            filelist = os.listdir(category_path)
            for file in filelist:
                fpath = os.path.join(category_path, file)
                if os.path.isfile(fpath):
                    filepaths.append(fpath)
                    labels.append(category)

# Create DataFrame from file paths and labels
data = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Split the data into train, validation, and test sets
train_df, dummy_df = train_test_split(data, train_size=0.8, shuffle=True, random_state=123, stratify=data['labels'])
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])

# Parameters for ImageDataGenerator
batch_size = 32
img_size = (200, 200)
channels = 3  # Assuming RGB images

# Data augmentation for training data
tr_gen = ImageDataGenerator(
    rotation_range=20,         
    width_shift_range=0.2,     
    height_shift_range=0.2,   
    shear_range=0.2,           
    zoom_range=0.2,            
    horizontal_flip=True,      
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'        
)

# Data generator for validation and test data
ts_gen = ImageDataGenerator()

# Create data generators
train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                      color_mode='rgb', shuffle=False, batch_size=batch_size)

# Plot data distribution
plt.pie([len(train_gen), len(valid_gen), len(test_gen)],
        labels=['train', 'validation', 'test'], autopct='%.1f%%', colors=['aqua', 'red', 'green'], explode=(0.05, 0, 0))
plt.show()
plt.savefig('dataset_pie.png')



print(train_gen.class_indices)
print(test_gen.class_indices)
print(valid_gen.class_indices)


base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
for layer in base_model.layers[-4:]: 
    layer.trainable = True
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'), 
      Dropout(0.5),
    Dense(4, activation='softmax')
])
optimizer = Adam(learning_rate=5e-5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=3, 
    restore_best_weights=True  
)
model.fit(
    train_gen,
    epochs=50,  
    validation_data=valid_gen,class_weight=class_weights_dict,
    callbacks=[early_stopping,reduce_lr]
)

model.save('bestmodelyet.keras')


dataset_dir = "cancerbrain"


import matplotlib.pyplot as plt
import numpy as np

# Retrieve the history from the model
history = model.history

epochs = range(1, len(history['accuracy']) + 1)
train_acc = history['accuracy']
val_acc = history['val_accuracy']
train_loss = history['loss']
val_loss = history['val_loss']

def annotate_min_max(x, y, ax=None, offset=0.05, fontsize=12):
    xmin = np.argmin(y)
    xmax = np.argmax(y)
    ymin = y[xmin]
    ymax = y[xmax]
    ax.annotate(f"Min\nEpoch: {x[xmin]}\nValue: {ymin:.4f}",
                xy=(x[xmin], ymin), xytext=(x[xmin] + 1, ymin - offset),
                arrowprops=dict(facecolor='red', shrink=0.05, headwidth=8, width=2),
                horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor='white', alpha=0.6))
    ax.annotate(f"Max\nEpoch: {x[xmax]}\nValue: {ymax:.4f}",
                xy=(x[xmax], ymax), xytext=(x[xmax] - 1, ymax + offset),
                arrowprops=dict(facecolor='green', shrink=0.05, headwidth=8, width=2),
                horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='green', facecolor='white', alpha=0.6))

plt.style.use('seaborn-darkgrid')

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(epochs, train_acc, label='Train Accuracy', marker='o', linestyle='--', color='b')
ax.plot(epochs, val_acc, label='Validation Accuracy', marker='o', linestyle='-', color='r')
ax.fill_between(epochs, train_acc, val_acc, color='gray', alpha=0.1)
ax.set_title('Model Accuracy over Epochs', fontsize=20)
ax.set_ylabel('Accuracy', fontsize=16)
ax.set_xlabel('Epoch', fontsize=16)
ax.set_xticks(epochs)
ax.legend(loc='lower right', fontsize=14, frameon=True, shadow=True, borderpad=1)
ax.grid(True, linestyle='--', linewidth=0.5)
annotate_min_max(epochs, val_acc, ax, offset=0.05, fontsize=12)
plt.tight_layout()
plt.show()

# Plotting training & validation loss values
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(epochs, train_loss, label='Train Loss', marker='o', linestyle='--', color='b')
ax.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='-', color='r')
ax.fill_between(epochs, train_loss, val_loss, color='gray', alpha=0.1)
ax.set_title('Model Loss over Epochs', fontsize=20)
ax.set_ylabel('Loss', fontsize=16)
ax.set_xlabel('Epoch', fontsize=16)
ax.set_xticks(epochs)
ax.legend(loc='upper right', fontsize=14, frameon=True, shadow=True, borderpad=1)
ax.grid(True, linestyle='--', linewidth=0.5)
annotate_min_max(epochs, val_loss, ax, offset=0.1, fontsize=12)
plt.tight_layout()
plt.show()



from sklearn.metrics import ConfusionMatrixDisplay

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=test_gen.class_indices)

# Display confusion matrix
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Changed File image path every time to see results

image_path = "cancerbrain/pituitary/Tr-pi_0019.jpg"  

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [200, 200])  
    image = tf.cast(image, tf.float32) / 255.0  
    return image.numpy()

image = preprocess_image(image_path)

def f(x):
    return model.predict(x)

masker_blur = shap.maskers.Image("blur(200,200)", (200, 200, 3))

explainer_blur = shap.Explainer(f, masker_blur)

predictions = model.predict(image[np.newaxis, :, :, :])
print("predictions",predictions)

class_names = ["glioma", "meningioma", "no tumor", "pituitary"]

top_class_index = np.argmax(predictions[0])
print("top class iindex ",top_class_index)

top_class_name = class_names[top_class_index]
top_class_probability = predictions[0][top_class_index]

print(f"Top class: {top_class_name} with probability {top_class_probability}")

top_4_indices = np.argsort(predictions[0])[::-1][:4]

print(f"Top 4 class indices: {top_4_indices}")
print(f"Top 4 class names: {[class_names[idx] for idx in top_4_indices]}")

shap_values_fine = explainer_blur(image[np.newaxis, :, :, :], max_evals=2000, outputs=top_4_indices)

print("Shape of SHAP values:", np.array(shap_values_fine.values).shape)

for i, class_index in enumerate(top_4_indices):
    shap.image_plot([shap_values_fine.values[0][:, :, :, i]], image, show=False)
    plt.title(f"SHAP Explanation for Class: {class_names[class_index]}")
    plt.show()



