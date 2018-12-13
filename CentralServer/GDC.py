"""
GoingDeeperClassifier.py
"""

import tensorflow as tf
import numpy as np


def load_graph_def(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # TODO: Get this info from the graph def input tensor:
    label_file = 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/output_labels.txt'
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = 'source_model'
    output_layer = 'InceptionV3/Predictions/Reshape_1'
    # Load the GraphDef protobuf file:
    graph_def_file_path = 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/trained_model/'
    # Load a test input image:
    # image_path = 'C:/Users/ccamp/Documents/GitHub/cs5450-final-project/CentralServer/Images/0/ari.jpg'
    image_path = 'C:/Users/ccamp/Documents/GitHub/cs5450-final-project/CentralServer/Images/0/CMC00007879.JPG'

    image_tensor = read_tensor_from_image_file(image_path, input_height, input_width, input_mean, input_std)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], graph_def_file_path)
        # input_name = 'import/' + input_layer
        # output_name = 'import/' + output_layer
        input_operation = sess.graph.get_operation_by_name('source_model/resized_input')
        output_operation = sess.graph.get_operation_by_name('retrain_ops/final_result')
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: image_tensor
        })
    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print('label: %s, %.2f%%' % (labels[i], results[i] * 100))

