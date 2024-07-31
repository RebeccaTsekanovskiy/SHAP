data_dir = 'cancerbrain'
filepaths = []
labels = []




folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)


    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        if fold == 'glioma':
            labels.append('glioma')


        elif fold == 'meningioma':
            labels.append('meningioma')


        elif fold == 'notumor':
            labels.append('notumor')


        elif fold == 'pituitary':
            labels.append('pituitary')


Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
data = pd.concat([Fseries, Lseries], axis=1)


data.head()
strat = data['labels']
train_df, dummy_df = train_test_split(data,  train_size= 0.8, shuffle= True, random_state= 123, stratify= strat)


strat = dummy_df['labels']
valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 123, stratify= strat)
print(train_df.head())


batch_size = 64
img_size = (200, 200)
channels = 4
img_shape = (img_size[0], img_size[1], channels)


tr_gen = ImageDataGenerator(
    rotation_range=20,          # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,      # Randomly shift images horizontally by up to 20% of the width
    height_shift_range=0.2,     # Randomly shift images vertically by up to 20% of the height
    shear_range=0.2,            # Apply random shearing transformations
    zoom_range=0.2,             # Randomly zoom in and out of images
    horizontal_flip=True,       # Randomly flip images horizontally
    brightness_range=[0.8, 1.2], # Randomly adjust brightness
    fill_mode='nearest'         # Fill in missing pixels with the nearest value
)


tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()


train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)


valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)


test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= batch_size)


plt.pie([len(train_gen), len(valid_gen), len(test_gen)],
        labels=['train', 'validation', 'test'], autopct='%.1f%%', colors=['aqua', 'red', 'green'], explode=(0.05, 0, 0))
plt.show()
plt.savefig('dataset_pie.png')


print(train_gen.class_indices)
print(test_gen.class_indices)
print(valid_gen.class_indices)


base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
for layer in base_model.layers[-4:]:  # Adjust the number of layers to fine-tune
    layer.trainable = True
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),  # Intermediate layer
      Dropout(0.5),
    Dense(4, activation='softmax')
])
optimizer = Adam(learning_rate=5e-5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,  # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore the weights of the best epoch
)
model.fit(
    train_gen,
    epochs=50,  # Set a larger number of epochs initially
    validation_data=valid_gen,class_weight=class_weights_dict,
    callbacks=[early_stopping,reduce_lr]
)

model.save('bestmodelyet.keras')




# Define the dataset directory
dataset_dir = "cancerbrain"


# Define the path to the image you want to explain
image_path = "cancerbrain/pituitary/Tr-pi_0017.jpg"
image_to_explain = ts_gen.standardize(image)


# Define the preprocess function
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [200, 200])  # Resize image to (224, 224)
    image = tf.cast(image, tf.float32) / 255.0   # Normalize pixel values
    return image.numpy()


# Preprocess the image
image = preprocess_image(image_path)


def f(x):
    tmp = x.copy()
    preprocess_input(tmp)
    return model(tmp)


# Create an Image masker for SHAP
masker_blur = shap.maskers.Image("blur(200,200)", shape=(200, 200, 3)) # Update masker shape


explainer_blur = shap.Explainer(f, masker_blur)


predictions = model.predict(image[np.newaxis, :, :, :])
top_4_indices = np.argsort(predictions[0])[::-1][:4]


shap_values_fine = explainer_blur(image[np.newaxis, :, :, :], max_evals=1000, outputs=top_4_indices)
print("Shape of SHAP values:", np.array(shap_values_fine.values).shape)




class_names = ["glioma", "meningioma", "no tumor", "pituitary"]


predictions = model.predict(image[np.newaxis, :, :, :])
# Select indices of top 4 classes (highest probabilities)
top_4_indices = np.argsort(predictions[0])[-4:]


# Plot the SHAP values for the top 4 classes, with labels
for i in range(len(top_4_indices)):
    sorted_index = top_4_indices[i]


    # Map the sorted index back to the actual class label
    class_index = np.where(np.argsort(predictions[0]) == sorted_index)[0][0]


    if class_index < len(class_names):
        shap.image_plot([shap_values_fine.values[0][:, :, :, i]], image, show=False)
        plt.title(f"SHAP Explanation for Class: {class_names[class_index]}")
        plt.show()
    else:
        print(f"Warning: Skipping invalid class index {class_index}")



import matplotlib.pyplot as plt
import numpy as np

# Retrieve the history from the model
history = model.history.history

# Extracting values for easier manipulation
epochs = range(1, len(history['accuracy']) + 1)
train_acc = history['accuracy']
val_acc = history['val_accuracy']
train_loss = history['loss']
val_loss = history['val_loss']

# Function to annotate the highest and lowest points
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

# Customizing the style
plt.style.use('seaborn-darkgrid')

# Plotting training & validation accuracy values
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



