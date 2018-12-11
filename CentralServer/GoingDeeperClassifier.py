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


def load_graph_def_file(graph_def_file_path):
    # Load the protobuff file from the disk and parse it to retrieve the un-serialized GraphDef object:
    with tf.gfile.GFile(graph_def_file_path, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
    # Import the graph_def into a new Graph and return it:
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='trained')
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


def create_bottleneck_file(bottleneck_path, image_data, sess, jpeg_data_tensor, decoded_image_tensor,
                           resized_input_tensor, bottleneck_tensor):
    try:
        bottleneck_value = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    except Exception as err:
        raise RuntimeError('Error during bottleneck creation while processing file %s (%s)' % (image_path,
                                                                     str(err)))
    bottleneck_string = str(bottleneck_value)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def calculate_bottleneck_value(sess, image_path, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    """
    calculate_bottleneck_value: Forward propagates the provided image through the original source network to produce the
        output tensor associated with the penultimate layer (pre softmax).
    :param sess: <tf.Session> The current active TensorFlow Session.
    :param image_path: <str> The file path to the original input image (used for debug purposes).
    :param image_data: <str> String of raw JPEG data.
    :param image_data_tensor: <?> Input data layer in the graph.
    :param decoded_image_tensor: <?> Output of initial image resizing and preprocessing.
    :param resized_input_tensor: <?> The input node of the source/recognition graph.
    :param bottleneck_tensor: <?> The penultimate node before the final softmax layer of the source/recognition graph.
    :return bottleneck_tensor_value: <numpy.ndarray> The result of forward propagating the provided image through the
        source/recognition graph.
    """
    tf.logging.info('Creating bottleneck for sample image ' + image_path)
    try:
        bottleneck_tensor_value = run_bottleneck_on_image(sess=sess, image_data=image_data,
                                                    image_data_tensor=image_data_tensor,
                                                    decoded_image_tensor=decoded_image_tensor,
                                                    resized_input_tensor=resized_input_tensor,
                                                    bottleneck_tensor=bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during bottleneck processing file %s (%s)' % (image_path,
                                                                     str(e)))
    return bottleneck_tensor_value


def add_jpeg_decoding(module_spec):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    return jpeg_data, resized_image


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load the GraphDef protobuf file:
    graph_def_file_path = 'C:/Users/ccamp/Documents/GitHub/HerbariumDeep/frameworks/TensorFlow/TFHub/tmp/intermediate_graphs/intermediate_10.pb'
    graph = load_graph_def_file(graph_def_file_path=graph_def_file_path)

    '''
    The image decoding subgraph wasn't exported with the original metagraph file (you can see this subgraph in 
        tensorboard under the re-train_ops_1 name scope). As a result, it is necessary to create an Image Decoding 
        graph here which performs resizing operations on the input image data.  
    '''
    image_decoding_graph = tf.Graph()
    # TODO: How to add jpeg_data_tensor and resized_image_tensor to the other graph?
    # Add jpeg decoding operations that were not exported with the original metagraph:
    module_spec = hub.load_module_spec('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
    jpeg_data_tensor, resized_image_tensor = add_jpeg_decoding(module_spec=module_spec)

    # Here we can get references to any Ops or Tensors we want.
    image_data_tensor = jpeg_data_tensor
    # image_data_tensor = graph.get_operation_by_name('trained/re-train_ops_1/DecodeJPGInput')
    # resized_image_tensor = graph.get_operation_by_name('trained/re-train_ops_1/ResizeBilinear')
    resized_input_tensor = graph.get_tensor_by_name('trained/source_model/resized_input:0')

    # bottleneck_tensor = graph.get_operation_by_name('trained/source_model/pre-trained_hub_module/bottleneck_pre_activation')
    # Tensor to store the output of the forward pass of the source model:
    # with tf.Session() as sess:
    #     model = hub.Module(module_spec)
    #     bottleneck_tensor = model(resized_input_tensor)

    retrain_op_bottleneck_input_tensor = graph.get_operation_by_name('trained/re-train_ops/input/BottleneckInputPlaceholder')

    retrain_ops_bottleneck_input = graph.get_tensor_by_name('trained/re-train_ops/input/BottleneckInputPlaceholder:0')
    softmax_op = graph.get_operation_by_name('trained/re-train_ops/final_result')

    # Load a test input image:
    image_path = 'C:/Users/ccamp/Documents/GitHub/cs5450-final-project/CentralServer/Images/0/ari.jpg'
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s' % image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Now we can use the graph we restored with the protobuf in a session (but keep in mind constant ops only):
    with tf.Session(graph=graph) as sess:
        '''
        Need to create a tensor to store the results of the forward pass through the initial source network:
        '''
        model = hub.Module(module_spec)
        bottleneck_tensor = model(resized_input_tensor)
        '''
        Note: No need to initialize or restore any tf.Variables, as there are none in this tf.Graph. Instead there are
            only hardcoded constants as a result of the conversion from tensors to constant ops prior to export.
        '''
        # Load the original source model to compute bottleneck vector for the source image:
        image_bottleneck_tensor = calculate_bottleneck_value(
            sess=sess, image_path=image_path, image_data=image_data,
            image_data_tensor=image_data_tensor, decoded_image_tensor=resized_image_tensor,
            resized_input_tensor=resized_input_tensor, bottleneck_tensor=bottleneck_tensor)

        # Forward pass the bottleneck tensor through the network:
        predict_proba = sess.run(softmax_op, feed_dict={retrain_op_bottleneck_input_tensor: image_bottleneck_tensor})
        print(predict_proba)


    # clf = GoingDeeperClassifier(checkpoint_dir='C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\')
    # classify_sample(image_path='C:/Users/ccamp/Documents/GitHub/cs5450-final-project/CentralServer/Images/0/ari.jpg')
