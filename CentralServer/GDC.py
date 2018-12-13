"""
GoingDeeperClassifier.py
"""

import tensorflow as tf
import tensorflow_hub as hub

import os
import numpy as np


def create_module_graph(module_spec):
    """
    create_module_graph: Creates a tensorflow graph from the provided TFHub module.
    source: https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
    :param module_spec: the hub.ModuleSpec for the image module being used.
    :returns:
        :return graph: The tf.Graph that was created.
        :return bottleneck_tensor: The bottleneck values output by the module.
        :return resized_input_tensor: The input images, resized as expected by the module.
        :return wants_quantization: A boolean value, whether the module has been instrumented with fake quantization
            ops.
    """
    # tf.reset_default_graph()
    # Define the receptive field in accordance with the chosen architecture:
    height, width = hub.get_expected_image_size(module_spec)
    # Create a new default graph:
    with tf.Graph().as_default() as graph:
        with tf.variable_scope('source_model'):
            # Create a placeholder tensor for input to the model.
            resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
            with tf.variable_scope('pre-trained_hub_module'):
                # Declare the model in accordance with the chosen architecture:
                m = hub.Module(module_spec, name='inception_v3_hub')
                # Create another place holder tensor to catch the output of the pre-activation layer:
                bottleneck_tensor = m(resized_input_tensor)
                # Give a name to this tensor:
                tf.identity(bottleneck_tensor, name='bottleneck_pre_activation')
                # This is a boolean flag indicating whether the module has been put through TensorFlow Light and optimized.
                # wants_quantization = any(node.op in FAKE_QUANT_OPS
                #                          for node in graph.as_graph_def().node)
    return graph, bottleneck_tensor, resized_input_tensor


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor, quantize_layer, is_training):
    """
    add_final_retrain_ops: Adds a new softmax and fully-connected layer for training and model evaluation. In order to
        use the TFHub model as a fixed feature extractor, we need to retrain the top fully connected layer of the graph
        that we previously added in the 'create_module_graph' method. This function adds the right ops to the graph,
        along with some variables to hold the weights, and then sets up all the gradients for the backward pass.

        The set up for the softmax and fully-connected layers is based on:
        https://www.tensorflow.org/tutorials/mnist/beginners/index.html
    :source https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
    :modified_by: Chris Campell
    :param class_count: An Integer representing the number of new classes we are trying to distinguish between.
    :param final_tensor_name: A name string for the final node that produces the fine-tuned results.
    :param bottleneck_tensor: The output of the main CNN graph (the specified TFHub module).
    :param quantize_layer: Boolean, specifying whether the newly added layer should be
        instrumented for quantization with TF-Lite.
    :param is_training: Boolean, specifying whether the newly add layer is for training
        or eval.
    :returns : The tensors for the training and cross entropy results, tensors for the
        bottleneck input and ground truth input, a reference to the optimizer for archival purposes and use in the
        hyper-string representation of this training run.
    """
    # The batch size
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size when ' \
                               'constructing fully-connected and softmax layers for fine-tuning.'

    # Tensor declarations:
    with tf.variable_scope('re-train_ops'):
        with tf.name_scope('input'):
            # Create a placeholder Tensor of same type as bottleneck_tensor to cache output from TFHub module:
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor,
                shape=[batch_size, bottleneck_tensor_size],
                name='BottleneckInputPlaceholder'
            )

            # Another placeholder Tensor to hold the true class labels
            ground_truth_input = tf.placeholder(
                tf.int64,
                shape=[batch_size],
                name='GroundTruthInput'
            )

        # Additional organization for TensorBoard:
        layer_name = 'final_retrain_ops'
        with tf.name_scope(layer_name):
            # Every layer has the following items:
            with tf.name_scope('weights'):
                # Output random values from truncated normal distribution:
                initial_value = tf.truncated_normal(
                    shape=[bottleneck_tensor_size, class_count],
                    stddev=0.001
                )
                layer_weights = tf.Variable(initial_value=initial_value, name='final_weights')
                # variable_summaries(layer_weights)

            with tf.name_scope('biases'):
                layer_biases = tf.Variable(initial_value=tf.zeros([class_count]), name='final_biases')
                # variable_summaries(layer_biases)

            # pre-activations:
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations_logits', logits)

        # This is the tensor that will hold the predictions of the fine-tuned (re-trained) model:
        final_tensor = tf.nn.softmax(logits=logits, name=final_tensor_name)

        # The tf.contrib.quantize functions rewrite the graph in place for
        # quantization. The imported model graph has already been rewritten, so upon
        # calling these rewrites, only the newly added final layer will be
        # transformed.
        if quantize_layer:
            if is_training:
                tf.contrib.quantize.create_training_graph()
            else:
                tf.contrib.quantize.create_eval_graph()

        # We will keep a histogram showing the distribution of activation functions.
        tf.summary.histogram('activations', final_tensor)

        # If this is an eval graph, we don't need to add loss ops or an optimizer.
        if not is_training:
            return None, None, bottleneck_input, ground_truth_input, final_tensor, 'No optimizer'

        with tf.name_scope('cross_entropy'):
            # What constitutes sparse in this case?:
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)

        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=CMD_ARG_FLAGS.learning_rate, momentum=0.9)
            # TODO: Can we make this not hard-coded? Trouble accessing the params passed to the optim at instantiation.
            if optimizer.get_name() == 'Momentum':
                optimizer_info = optimizer.get_name() + '{momentum=%.2f}' % optimizer._momentum
            else:
                optimizer_info = optimizer.get_name() + '{%s}' % (optimizer.get_slot_names())
                # optimizer_info = {slot_name: slot_value for slot_name, slot_value in zip(optimizer.get_slot_names(), optimizer.'_'.join(...)}
                # optimizer_info = optimizer.get_name() + '{%s}' % optimizer.variables()
                # optimizer_info = optimizer.get_name() + '{%s=%.2f}' % (optimizer.get_slot_names()[0], optimizer._momentum)
            train_step = optimizer.minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor, optimizer_info


class GoingDeeperClassifier:

    def __init__(self, checkpoint_dir):
        self.module_spec = hub.load_module_spec('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
        # self.class_count = ?
        eval_graph, bottleneck_tensor, _ = create_module_graph(self.module_spec)
        tf.logging.info(msg='Defined computational graph from the tensorflow hub module spec.')
        eval_sess = tf.Session(graph=eval_graph)
        class_count = 9
        with eval_graph.as_default():
            # Add the new layer for exporting.
            wants_quantization = False
            final_tensor_name = 'output_tensor'
            (_, _, bottleneck_input,
             ground_truth_input, final_tensor, optimizer_info) = add_final_retrain_ops(
                class_count, final_tensor_name, bottleneck_tensor,
                wants_quantization, is_training=False)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            tf.train.Saver().restore(eval_sess, ckpt.model_checkpoint_path)
            # test_bottlenecks, test_ground_truth, test_filenames = get_random_cached_bottlenecks(...)
            # (eval_sess, _ , bottleneck_input, ground_truth_input, evaluation_step, prediction) = build_eval_session(self.module_spec, class_count)
            # test_accuracy, predictions = eval_sess.run(
            #     [evaluation_step, prediction],
            #     feed_dict = {
            #         bottleneck_input: test_bottlenecks,
            #         ground_truth_input: test_ground_truth
            #     }
            # )
            # print()


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/', 'intermediate_graphs/intermediate_10.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # graph_def_path = 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/summaries/session_graph_def.meta'
        _ = tf.import_graph_def(graph_def, name='saved_model')


def classify_sample(image_path):
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s' % image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Create a graph from the saved GraphDef:
    create_graph()

    with tf.Session() as sess:
        print([n.name for n in sess.graph.as_graph_def().node])
        # tf.train.Saver().restore(sess, 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/_retrain_checkpoint')
        # Can't use sess.graph.get_tensor_by_name() as they are no longer tensors, they have been converted to constant ops for serialization.
        # softmax_tensor = sess.graph.get_tensor_by_name('saved_model/re-train_ops/final_result')
        softmax_const_op = sess.graph.get_operation_by_name('saved_model/re-train_ops/final_result')

        # Loading the injected placeholder:
        # for tensor in sess.graph.as_graph_def().node:
        #     print(tensor.name)
        # [n.name for n in sess.graph.as_graph_def().node]

        # input_placeholder = sess.graph.get_tensor_by_name('saved_model/re-train_ops/input/BottleneckInputPlaceholder')
        input_placeholder_const_op = sess.graph.get_operation_by_name('saved_model/re-train_ops/input/BottleneckInputPlaceholder')
        probas = sess.run(softmax_const_op, {input_placeholder_const_op: image_data})
        print(probas)
        # softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        # predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    # predictions = np.squeeze(predictions)

#
# def load_graph_def(graph_def_file_path):
#     # Load the protobuff file from the disk and parse it to retrieve the un-serialized GraphDef object:
#     with tf.gfile.GFile(graph_def_file_path, 'rb') as fp:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(fp.read())
#     # Import the graph_def into a new Graph and return it:
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def)
#     return graph


def load_graph_def(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      decoded_image_tensor: Output of initial image resizing and preprocessing.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: Layer before the final softmax.

    Returns:
      Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
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
    # graph_def_file_path = 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/intermediate_graphs/intermediate_10.pb'
    # graph_def_file_path = 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/trained_model/saved_model.pb'
    graph_def_file_path = 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/trained_model/'
    # graph_def_file_path = 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/summaries/session_graph_def.meta'
    # graph = load_graph_def(graph_def_file_path)
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
    # print(sum(results))

    '''
    The image decoding subgraph wasn't exported with the original metagraph file (you can see this subgraph in 
        tensorboard under the re-train_ops_1 name scope). As a result, it is necessary to create an Image Decoding 
        graph here which performs resizing operations on the input image data.  
    '''
    # image_decoding_graph = tf.Graph()
    # TODO: How to add jpeg_data_tensor and resized_image_tensor to the other graph?
    # Add jpeg decoding operations that were not exported with the original metagraph:
    # module_spec = hub.load_module_spec('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
    # jpeg_data_tensor, resized_image_tensor = add_jpeg_decoding(module_spec=module_spec)

    # Here we can get references to any Ops or Tensors we want.
    # image_data_tensor = jpeg_data_tensor
    # image_data_tensor = graph.get_operation_by_name('trained/re-train_ops_1/DecodeJPGInput')
    # resized_image_tensor = graph.get_operation_by_name('trained/re-train_ops_1/ResizeBilinear')
    # resized_input_tensor = graph.get_tensor_by_name('trained/source_model/resized_input:0')

    # bottleneck_tensor = graph.get_operation_by_name('trained/source_model/pre-trained_hub_module/bottleneck_pre_activation')
    # Tensor to store the output of the forward pass of the source model:
    # with tf.Session() as sess:
    #     model = hub.Module(module_spec)
    #     bottleneck_tensor = model(resized_input_tensor)

    # retrain_op_bottleneck_input_tensor = graph.get_operation_by_name('trained/re-train_ops/input/BottleneckInputPlaceholder')

    # retrain_ops_bottleneck_input = graph.get_tensor_by_name('trained/re-train_ops/input/BottleneckInputPlaceholder:0')
    # softmax_op = graph.get_operation_by_name('trained/re-train_ops/final_result')
