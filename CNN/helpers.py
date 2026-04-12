import os
import matplotlib.pyplot as plt
import tensorflow as tf


def get_dataset_dirs(dataset_dir: str):
  """Return train, validation, and test directory paths."""
  train_dir = os.path.join(dataset_dir, "training")
  valid_dir = os.path.join(dataset_dir, "validation")
  test_dir = os.path.join(dataset_dir, "test")
  return train_dir, valid_dir, test_dir


def get_class_names(train_dir: str):
  """Read class folder names from the training directory."""
  class_names = sorted(
      [
          folder_name
          for folder_name in os.listdir(train_dir)
          if os.path.isdir(os.path.join(train_dir, folder_name))
      ]
  )
  return class_names


def load_datasets(dataset_dir: str, img_size=(64, 64), batch_size: int = 32, seed: int = 42):
  train_dir, valid_dir, test_dir = get_dataset_dirs(dataset_dir)
  class_names = get_class_names(train_dir)

  train_dataset = tf.keras.utils.image_dataset_from_directory(
      train_dir,
      labels="inferred",
      label_mode="int",
      class_names=class_names,
      image_size=img_size,
      batch_size=batch_size,
      shuffle=True,
      seed=seed,
  )

  valid_dataset = tf.keras.utils.image_dataset_from_directory(
      valid_dir,
      labels="inferred",
      label_mode="int",
      class_names=class_names,
      image_size=img_size,
      batch_size=batch_size,
      shuffle=False,
  )

  test_dataset = tf.keras.utils.image_dataset_from_directory(
      test_dir,
      labels="inferred",
      label_mode="int",
      class_names=class_names,
      image_size=img_size,
      batch_size=batch_size,
      shuffle=False,
  )

  autotune = tf.data.AUTOTUNE
  train_dataset = train_dataset.prefetch(buffer_size=autotune)
  valid_dataset = valid_dataset.prefetch(buffer_size=autotune)
  test_dataset = test_dataset.prefetch(buffer_size=autotune)

  return train_dataset, valid_dataset, test_dataset, class_names


def plot_history(history):
  acc = history.history["accuracy"]
  val_acc = history.history["val_accuracy"]
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  plt.figure(figsize=(8, 5))
  plt.plot(acc, label="train accuracy")
  plt.plot(val_acc, label="val accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.title("Training and Validation Accuracy")
  plt.show()

  plt.figure(figsize=(8, 5))
  plt.plot(loss, label="train loss")
  plt.plot(val_loss, label="val loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.title("Training and Validation Loss")
  plt.show()
