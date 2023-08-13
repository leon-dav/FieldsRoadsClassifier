import tensorflow as tf


def data_loader(
    train_folder: str, unannotated_folder: str, batch_size: int, img_size: tuple
) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list):
    """Generate and return 4 TensorFlow datasets and the class names

    train_folder: str -- path to the folder containing the training images. The directory structure must be compatible with tf.keras.utils.image_dataset_from_directory
    unannotated_folder: str -- path to the folder containing the unannotated images. The directory structure must be compatible with tf.keras.utils.image_dataset_from_directory
    batch_size: int
    img_size: tuple -- (int, int)
    """
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_folder,
        shuffle=True,
        subset="training",
        validation_split=0.3,
        seed=42,
        batch_size=batch_size,
        image_size=img_size,
    )

    # extract class names
    class_names = train_dataset.class_names

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_folder,
        shuffle=True,
        validation_split=0.3,
        seed=42,
        subset="validation",
        batch_size=batch_size,
        image_size=img_size,
    )

    qualitatif_test_dataset = tf.keras.utils.image_dataset_from_directory(
        unannotated_folder, labels=None, image_size=img_size
    )

    # take images for our test dataset
    test_dataset = train_dataset.take(2)
    train_dataset = train_dataset.skip(2)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        qualitatif_test_dataset,
        class_names,
    )


def get_model(feature_extractor: str, img_size: tuple) -> tf.keras.Model:
    """Return a Keras model with the given pre-trained feature extractor.

    feature_extractor: str -- the backbone to use as feature extractor
    img_size: tuple -- (int, int)
    """
    if feature_extractor == "mobilenetv2":
        extractor = tf.keras.applications.MobileNetV2
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif feature_extractor == "convnext":
        extractor = tf.keras.applications.ConvNeXtBase
        preprocess_input = tf.keras.applications.convnext.preprocess_input
    elif feature_extractor == "densenet":
        extractor = tf.keras.applications.DenseNet121
        preprocess_input = tf.keras.applications.densenet.preprocess_input
    elif feature_extractor == "efficientnetb0":
        extractor = tf.keras.applications.EfficientNetB0
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError("Feature extractor not found")

    # Use a pretrained model as our feature extractor
    base_model = extractor(
        input_shape=img_size + (3,), include_top=False, weights="imagenet"
    )

    # don't train our feature extractor
    base_model.trainable = False

    # preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # use data augmentation
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ]
    )

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # Dense prediction layer
    prediction_layer = tf.keras.layers.Dense(1)

    # build model
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model
