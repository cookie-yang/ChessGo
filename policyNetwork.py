import tensorflow as tf
import numpy as np
import os
import sys


def cnn_model_fn(features,labels,mode):
    #Input layer

    input_layer = tf.reshape(features["x"], [-1, 8, 8, 6])


    #Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters = 192,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
        )
    dp1 = tf.layers.dropout(conv1,rate=0.4)
    bn1 = tf.layers.batch_normalization(dp1)

    tf.get_variable
    #Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=bn1,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    dp2 = tf.layers.dropout(conv2,rate=0.4)
    bn2 = tf.layers.batch_normalization(dp2)

    #Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=bn2,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    dp3 = tf.layers.dropout(conv3,rate=0.4)
    bn3 = tf.layers.batch_normalization(dp3)

    #Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=bn3,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    dp4 = tf.layers.dropout(conv4,rate=0.4)
    bn4 = tf.layers.batch_normalization(dp4)

    #Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=bn4,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    dp5 = tf.layers.dropout(conv5,rate=0.4)
    bn5 = tf.layers.batch_normalization(dp5)

    #Convolutional Layer #6
    conv6 = tf.layers.conv2d(
        inputs=bn5,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    dp6 = tf.layers.dropout(conv6,rate=0.4)
    bn6 = tf.layers.batch_normalization(dp6)
    #Convolutional Layer #7
    conv7 = tf.layers.conv2d(
        inputs=bn6,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    dp7 = tf.layers.dropout(conv7,rate=0.4)
    bn7 = tf.layers.batch_normalization(dp7)

    #Convolutional Layer #8
    conv8 = tf.layers.conv2d(
        inputs=bn7,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    dp8 = tf.layers.dropout(conv8,rate=0.4)
    bn8 = tf.layers.batch_normalization(dp8)
    #Convolutional Layer #9
    
    
    # conv13 = tf.layers.conv2d(
    #     inputs=conv12,
    #     filters=1,
    #     kernel_size=[1, 1],
    #     use_bias=False,
    #     activation=newBiasAdd
    # )

    #flatten tensor to create softmax layer
    faltten_conv8 = tf.reshape(bn8, [-1, 8*8*192])

    #Logits Layer
    abstractMoves = tf.layers.dense(inputs=faltten_conv8, units=378)


    #Prediction
    predictions = {
        "classes": tf.argmax(input=abstractMoves, axis=1),
        "probabilities": tf.nn.softmax(abstractMoves, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=378)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=abstractMoves)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
   
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Add evaluation metrics (for EVAL mode)
   
     # Configure the Training Op (for TRAIN mode)
    learningRate = tf.train.exponential_decay(learning_rate=0.003,global_step=tf.train.get_global_step(),decay_steps=600000,decay_rate=0.5)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learningRate,momentum=0.9,use_nesterov=True)
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    tf.summary.scalar("my_accuracy",accuracy[1])
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



def main(unused_argv):
    #read data
    DATAPATH = "../data/policynetwork/data_policy_l_232_float32.npz"
    if DATAPATH != None:
        with np.load(DATAPATH) as data:
            data_features = data["feature"]
            data_labels = data["label"]
        assert data_features.shape[0] == data_labels.shape[0]

    # data_features, data_labels = createData("../data/data_s")
    num_data = data_features.shape[0]
    train_data_features = data_features[:int(0.995*num_data)]
    test_data_features = data_features[int(0.995*num_data):]
    train_data_labels = data_labels[:int(0.995*num_data)]
    test_data_labels = data_labels[int(0.995*num_data):]

    #create estimator
    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="../model/policynetwork_eight_layer_alldata_1024batch_100epoch_"
        )
    #create log
    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=1)
    # with tf.Session() as sees:
        # sees.run(iterator.initializer,feed_dict={features_placeholder:features,labels_placeholder:labels})
        # print(sees.run(data))
        # return
        #    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data_features},
        y=train_data_labels,
        batch_size=1024,
        num_epochs=None,
        shuffle=True
    )

    cnn_classifier.train(
        input_fn=train_input_fn,
        steps = 2000000)

    #evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_data_features},
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

