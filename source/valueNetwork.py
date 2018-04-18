import tensorflow as tf
import numpy as np
import os
import sys


def cnn_model_fn(features, labels, mode):
    #Input layer

    input_layer = tf.reshape(features["x"], [-1, 8, 8, 7])

    #Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=192,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu
    )
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    pad1 = tf.pad(conv1, paddings, "CONSTANT")
    #Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pad1,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #6
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #7
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #8
    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #9
    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #10
    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #11
    conv11 = tf.layers.conv2d(
        inputs=conv10,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    #Convolutional Layer #12
    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=1,
        kernel_size=[1, 1],
        use_bias=False,
        activation=tf.nn.relu
    )

    #flatten tensor to create softmax layer
    faltten_conv13 = tf.reshape(conv13, [-1, 8*8*1])

    #full connected layer
    full_connect14 = tf.layers.dense(inputs=faltten_conv13, units=256,activation=tf.nn.relu)

    #output layer
    output = tf.layers.dense(inputs=full_connect14,units=1,activation=tf.nn.tanh)
    #Prediction
    predictions = {
        "probability": output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    print(labels)
    print(output)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,logits=output)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["probability"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    #read data
    DATAPATH = "../data/data_value_s.npz"
    if DATAPATH != None:
        with np.load(DATAPATH) as data:
            data_features = data["feature"]
            data_labels = data["label"]
        assert data_features.shape[0] == data_labels.shape[0]

    # data_features, data_labels = createData("../data/data_s")
    num_data = data_features.shape[0]
    train_data_features = data_features[:int(0.9*num_data)]
    test_data_features = data_features[int(0.9*num_data):]
    train_data_labels = data_labels[:int(0.9*num_data)]
    test_data_labels = data_labels[int(0.9*num_data):]

    #create estimator
    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="../model/valuenetwork"
    )
    #create log
    # tensors_to_log = {"probabilities": "tanh-tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=1)
    # with tf.Session() as sees:
    # sees.run(iterator.initializer,feed_dict={features_placeholder:features,labels_placeholder:labels})
    # print(sees.run(data))
    # return
    #
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data_features},
        y=train_data_labels,
        batch_size=2,
        num_epochs=50,
        shuffle=True
    )

    cnn_classifier.train(
        input_fn=train_input_fn,
        steps=400)

    #evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data_features},
        y=test_data_labels,
        num_epochs=1,
        shuffle=False
    )

    result = cnn_classifier.evaluate(
        input_fn=eval_input_fn
    )
    print(result)




if __name__ == "__main__":
    tf.app.run()
