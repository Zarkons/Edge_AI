import tensorflow as tf
import numpy as np

import math
import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 100, "Number of epochs to train.")
flags.DEFINE_string("save_dir", "./", "Directory to save the trained model.")
flags.DEFINE_boolean("save_tf_model", False,
                     "store the original unconverted tf model.")

def generate_random_input_tensor():
    random_input = np.random.normal(loc=0.5, scale=0.05, size=(1, 3)).astype(np.float32)
    return random_input

def get_data():
    # Generates a clean (1000, 3) array instantly
    train_data = np.random.normal(loc=0.5, scale=0.05, size=(1000, 3)).astype(np.float32)
    val_data = np.random.normal(loc=0.5, scale=0.05, size=(200, 3)).astype(np.float32)
    return train_data, val_data


def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential()

    # First layer takes a 3 inputs and feeds it through 2 "neurons". The
    # neurons decide whether to activate based on the 'relu' activation function.
    model.add(tf.keras.layers.Dense(2, activation='relu', input_shape=(3, )))

    # Final layer are 3 neurons, since we want to output 3 values
    model.add(tf.keras.layers.Dense(3))

    # Compile the model using the standard 'adam' optimizer and the mean squared
    # error or 'mse' loss function for regression.
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

def train_model(epochs, normal_data, val_data):
    model = create_model()
    model.fit(normal_data, normal_data, epochs=epochs, batch_size=32, verbose=0, validation_data=(val_data, val_data))

    if FLAGS.save_tf_model:
        model.save(FLAGS.save_dir, save_format="tf")
        logging.info("TF model saved to %s", FLAGS.save_dir)

    return model

def convert_tflite_model_quantized(model):
    """Convert the save TF model to quantized tflite model, then save it as .tflite flatbuffer format
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    """Tells the converter to quantize the weights and activations of the model to 8-bits. Permanently
    squezes the weights and activations to 8-bits.
    """
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for _ in range(100):
            data = np.random.normal(loc=0.5, scale=0.05, size=(1, 3)).astype(np.float32)
            yield [data]

    """"Without this we have only hibryd model,
    """
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()
    return tflite_model

def convert_tflite_model(model):
    """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model

def save_tflite_model(tflite_model, save_dir, model_name):
    """save the converted tflite model
    Args:
        tflite_model (binary): the converted model in serialized format.
        save_dir (str): the save directory
        model_name (str): model name to be saved
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_name)
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    logging.info("Tflite model saved to %s", save_dir)

def plot_training_history(history):
    
    # Plot training & validation loss values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def run_inference(interpreter, predictions_list, input_index, output_index, input_data):

    # Run inference
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    predictions_list.append(output)

def plot_predictions(prediction, prediction_quantized):

    plt.figure(figsize=(8, 4))
    plt.plot(prediction.flatten(), label='Original Model Prediction', marker='o')
    plt.plot(prediction_quantized.flatten(), label='Quantized Model Prediction', marker='x')
    plt.title('Model Predictions Comparison')
    plt.xlabel('Output Index')
    plt.ylabel('Predicted Value')
    plt.legend()
    plt.grid()
    plt.show()


def calculate_model_size(model, model_quantized):

    model_size = os.path.getsize(model)
    logging.info("Model size: %.2f B", model_size)

    model_quantized_size = os.path.getsize(model_quantized)
    logging.info("Quantized Model size: %.2f B", model_quantized_size)

    logging.info("Size reduction: %.2f%%", (model_size - model_quantized_size) / model_size * 100)
def main(_):
    train_data, val_data = get_data()
    
    trained_model = train_model(FLAGS.epochs, train_data, val_data)

    # plot_training_history(trained_model.history)

    # Convert the model to .tflite memory buffer and begin with TensorFlow Lite Micro Interpreter
    tflite_model = convert_tflite_model(trained_model)
    tflite_quantized_model = convert_tflite_model_quantized(trained_model)

    # Load the tflite model into an interpreter to verify it's valid and can be executed
    model_interpreter = tf.lite.Interpreter(model_content=tflite_model)
    model_quantized_interpreter = tf.lite.Interpreter(model_content=tflite_quantized_model)

    # # Allocate tensors to prepare the model for inference. This step is necessary before invoking the interpreter.
    model_interpreter.allocate_tensors()
    model_quantized_interpreter.allocate_tensors()

    model_input_index = model_interpreter.get_input_details()[0]['index']
    model_quantized_input_index = model_quantized_interpreter.get_input_details()[0]['index']
    model_output_index = model_interpreter.get_output_details()[0]['index']
    model_quantized_output_index = model_quantized_interpreter.get_output_details()[0]['index']

    model_predictions = []
    model_quantized_predictions = []

    for data in train_data:
        input_tensor_data = np.expand_dims(data, axis=0).astype(np.float32)
        
        # Run inference on the original model
        run_inference(model_interpreter, model_predictions, model_input_index, model_output_index, input_tensor_data)

        # Run inference on the quantized model
        run_inference(model_quantized_interpreter, model_quantized_predictions, model_quantized_input_index, model_quantized_output_index, input_tensor_data)

    #plot_predictions(model_predictions[0], model_quantized_predictions[0])

    # Convert the list of predictions to a numpy array for easier manipulation
    quantized_pred_matrix = np.vstack(model_quantized_predictions)

    # Now you can safely subtract and calculate MSE across axis 1
    mse_per_sample = np.mean(np.power(train_data - quantized_pred_matrix, 2), axis=1)

    # Set the threshold
    anomaly_threshold = np.max(mse_per_sample)
    


    # Check if executed via 'bazel run'
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    
    if workspace_dir:
        # Resolve path directly to your project's physical source folder
        actual_save_dir = os.path.join(workspace_dir, "learning/example/anomaly_detection")
    else:
        # Fallback default path if run outside of Bazel
        actual_save_dir = FLAGS.save_dir

    # Save the model to the correctly resolved folder
    save_tflite_model(
        tflite_model,
        save_dir=actual_save_dir,
        model_name="anomaly_detection.tflite"
    )
    save_tflite_model(
        tflite_quantized_model,
        save_dir=actual_save_dir,
        model_name="anomaly_detection_quantized.tflite"
    )

    calculate_model_size(
        model=os.path.join(actual_save_dir, "anomaly_detection.tflite"),
        model_quantized=os.path.join(actual_save_dir, "anomaly_detection_quantized.tflite")
    )
    print(f"Your Quantized Anomaly Threshold is: {anomaly_threshold}")
    


if __name__ == "__main__":
    app.run(main)   