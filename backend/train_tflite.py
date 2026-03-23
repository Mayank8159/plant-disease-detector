import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Plant Disease model and export TFLite")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="plantvillage dataset/color",
        help="Directory containing class subfolders of images",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="model.tflite")
    parser.add_argument("--labels-out", type=str, default="labels.txt")
    return parser.parse_args()


def build_datasets(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    val_split: float,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        validation_split=val_split,
        subset="training",
        seed=seed,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        shuffle=True,
    )

    class_names = train_ds.class_names

    def augment(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = tf.image.random_flip_left_right(images)
        return images, labels

    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Keep memory usage low: no dataset caching, small prefetch buffer.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


def build_model(num_classes: int, img_size: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def representative_dataset(data_dir: Path, img_size: int, limit: int = 200):
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(data_dir.rglob(ext))

    image_files = image_files[:limit]

    for img_path in image_files:
        image = tf.keras.utils.load_img(img_path, target_size=(img_size, img_size))
        array = tf.keras.utils.img_to_array(image)
        array = np.expand_dims(array, axis=0).astype(np.float32) / 255.0
        yield [array]


def export_tflite(model: tf.keras.Model, model_out: Path, data_dir: Path, img_size: int) -> None:
    saved_model_dir = model_out.parent / "_saved_model_export"
    if saved_model_dir.exists():
        tf.io.gfile.rmtree(str(saved_model_dir))

    model.export(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(data_dir, img_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    model_out.write_bytes(tflite_model)

    tf.io.gfile.rmtree(str(saved_model_dir))


def main() -> None:
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    data_dir = Path(args.data_dir).resolve()
    model_out = Path(args.model_out).resolve()
    labels_out = Path(args.labels_out).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    print(f"Using dataset: {data_dir}")
    train_ds, val_ds, class_names = build_datasets(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    print(f"Detected {len(class_names)} classes")
    model = build_model(num_classes=len(class_names), img_size=args.img_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f} | loss: {val_loss:.4f}")
    print(f"Best training accuracy: {max(history.history.get('accuracy', [0.0])):.4f}")

    labels_out.write_text("\n".join(class_names), encoding="utf-8")
    export_tflite(model=model, model_out=model_out, data_dir=data_dir, img_size=args.img_size)

    print(f"Saved TFLite model to: {model_out}")
    print(f"Saved labels to: {labels_out}")


if __name__ == "__main__":
    main()
