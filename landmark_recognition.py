import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


BASE_PATH = "/kaggle/input/competitions/landmark-recognition-2021"

print("Files and folders:")
print(os.listdir(BASE_PATH))


train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))

print("\nDataset Shape:", train_df.shape)

print("\nFirst 5 rows:")
print(train_df.head())


print("\nNumber of unique landmarks:",train_df["landmark_id"].nunique())

print("\nTop 10 Landmark IDs:")
print(train_df["landmark_id"].value_counts().head(10))


top_landmarks = (train_df["landmark_id"].value_counts().head(5))

print("\nSelected Landmark IDs:")
print(top_landmarks)


selected_ids = top_landmarks.index.tolist()


sample_df = train_df[train_df["landmark_id"].isin(selected_ids)].copy()


print("\nTotal selected images:", len(sample_df))

print("\nImages per landmark:")
print(sample_df["landmark_id"].value_counts())


MAX_IMAGES_PER_CLASS = 500

sample_list = []


for landmark_id, group in sample_df.groupby("landmark_id"):

    selected = group.sample(n=min(len(group), MAX_IMAGES_PER_CLASS),random_state=42)

    sample_list.append(selected)


small_df = pd.concat(sample_list,ignore_index=True)


print("\nTotal images selected:", len(small_df))

print("\nImages per landmark:")
print(small_df["landmark_id"].value_counts())


def get_image_path(image_id):

    return os.path.join(BASE_PATH,"train",image_id[0],image_id[1],image_id[2],image_id + ".jpg")


small_df["image_path"] = (small_df["id"].apply(get_image_path))


print("\nFirst 5 image paths:")
print(small_df[["id", "landmark_id", "image_path"]].head())


existing_images = (small_df["image_path"].apply(os.path.exists).sum())


print("\nExisting images:", existing_images)

print("Total selected images:",len(small_df))


label_encoder = LabelEncoder()


small_df["label"] = (label_encoder.fit_transform(small_df["landmark_id"]))


print("\nNumber of classes:",len(label_encoder.classes_))


train_data, val_data = train_test_split(small_df,test_size=0.2,random_state=42,stratify=small_df["label"])


print("\nTraining images:",len(train_data))

print("Validation images:",len(val_data))


IMG_SIZE = 224
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)


val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True
)


val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)


print("\nTrain Generator Ready!")
print("Validation Generator Ready!")


base_model = MobileNetV2(weights="imagenet",include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3))


base_model.trainable = False


x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256,activation="relu")(x)

x = Dropout(0.3)(x)


output = Dense(len(label_encoder.classes_),activation="softmax")(x)


model = Model(inputs=base_model.input,outputs=output)


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


model.summary()


early_stopping = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)


model_checkpoint = ModelCheckpoint("best_landmark_model.keras",monitor="val_accuracy",save_best_only=True,mode="max")


history = model.fit(train_generator,validation_data=val_generator,epochs=10,callbacks=[early_stopping,model_checkpoint])


val_loss, val_accuracy = model.evaluate(val_generator)


print("\nValidation Loss:", val_loss)

print("Validation Accuracy:",val_accuracy)


plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"],label="Training Accuracy")

plt.plot(history.history["val_accuracy"],label="Validation Accuracy")

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"],label="Training Loss")

plt.plot(history.history["val_loss"],label="Validation Loss")

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


sample_row = val_data.iloc[0]

image_path = sample_row["image_path"]

true_label = sample_row["label"]


img = image.load_img(image_path,target_size=(IMG_SIZE, IMG_SIZE))


img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array,axis=0)

img_array = img_array / 255.0


prediction = model.predict(img_array)


predicted_label = np.argmax(prediction[0])


confidence = (np.max(prediction[0]) * 100)


true_landmark = (label_encoder.inverse_transform([true_label])[0])


predicted_landmark = (label_encoder.inverse_transform([predicted_label])[0])


plt.figure(figsize=(6, 6))

plt.imshow(img)

plt.axis("off")

plt.title(
    f"True Landmark: {true_landmark}\n"
    f"Predicted Landmark: {predicted_landmark}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.show()


val_predictions = model.predict(val_generator)


predicted_labels = np.argmax(val_predictions,axis=1)


true_labels = (val_data["label"].values)


print("\nClassification Report:\n")


print(classification_report(true_labels,predicted_labels,target_names=[str(x)for x in label_encoder.classes_]))


cm = confusion_matrix(true_labels,predicted_labels)


plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.colorbar()


classes = [str(x)for x in label_encoder.classes_]


plt.xticks(range(len(classes)),classes,rotation=45)


plt.yticks(range(len(classes)),classes)


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(j,i,str(cm[i, j]),ha="center",va="center")


plt.tight_layout()

plt.show()


model.save("landmark_recognition_model.keras")


print("\nModel saved successfully!")


loaded_model = load_model("landmark_recognition_model.keras")


print("Saved model loaded successfully!")
