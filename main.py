import matplotlib.pyplot as plt
import tensorflow as tf
from utils import data_loader, get_model


IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 300
PATIENCE = 5

###################################################################
# Load data and create 4 datasets: train, validation, test
###################################################################

train_ds, val_ds, test_ds, qualitatif_test_ds, class_names = data_loader(
    "dataset/train", "dataset/test_images", BATCH_SIZE, IMG_SIZE,
)

###################################################################
# Create model
###################################################################

model = get_model("mobilenetv2", IMG_SIZE)

# compile using adam optimizer and BinaryCrossentropy loss
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# monitor validation loss for early stopping
loss_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)

# train the model
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[loss_callback],
)


###################################################################
# Results
###################################################################

# Total model params
print(f"Total model parameters: {str(model.count_params())}")

# Test accuracy
loss, accuracy = model.evaluate(test_ds)
print("Test accuracy :", accuracy)


# plot accuracy and loss evolution
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")

plt.savefig("accuracy_loss_evolution.png")
# plt.show()


# plot our qualitative test dataset with the predicted label
# Retrieve a batch of images from the test set
image_batch = qualitatif_test_ds.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print("Predictions:\n", predictions.numpy())

plt.figure(figsize=(5, 3))
for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")

plt.savefig("image_predictions.png")
# plt.show()
