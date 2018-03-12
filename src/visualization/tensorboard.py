import os
import keras
import tensorflow as tf


LOG_DIR = "/tmp/tensorboard"
os.makedirs(LOG_DIR, exist_ok=True)
OUTPUT_MODEL_FILE_NAME = os.path.join(LOG_DIR, "tf.ckpt")


def visualize_embedding(embedding_matrix, tensor_name, metadata):
    tf.Variable(embedding_matrix, name=tensor_name)

    metadata_path = os.path.join(LOG_DIR, tensor_name)
    metadata.to_csv(metadata_path, sep="\t", index=False)

    saver = tf.train.Saver()
    saver.save(keras.backend.get_session(), OUTPUT_MODEL_FILE_NAME)

    summary_writer = tf.summary.FileWriter(LOG_DIR)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor_name
    embedding.metadata_path = metadata_path

    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(
        summary_writer, config)
