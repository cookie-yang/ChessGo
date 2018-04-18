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
    bn1 = tf.layers.batch_normalization(conv1)
    # paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
    # pad1 = tf.pad(conv1,paddings,"CONSTANT")
    #Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=bn1,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    bn2 = tf.layers.batch_normalization(conv2)

    #Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=bn2,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    bn3 = tf.layers.batch_normalization(conv3)

    #Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=bn3,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    bn4 = tf.layers.batch_normalization(conv4)

    #Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=bn4,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    bn5 = tf.layers.batch_normalization(conv5)

    #Convolutional Layer #6
    conv6 = tf.layers.conv2d(
        inputs=bn5,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    bn6 = tf.layers.batch_normalization(conv6)
    #Convolutional Layer #7
    conv7 = tf.layers.conv2d(
        inputs=bn6,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    bn7 = tf.layers.batch_normalization(conv7)

    #Convolutional Layer #8
    conv8 = tf.layers.conv2d(
        inputs=bn7,
        filters=192,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    bn8 = tf.layers.batch_normalization(conv8)
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
    abstractMoves = tf.layers.dense(inputs=faltten_conv8, units=138)

    #Prediction
    predictions = {
        "classes": tf.argmax(input=abstractMoves, axis=1),
        "probabilities": tf.nn.softmax(abstractMoves, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=138)
    print(onehot_labels)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=abstractMoves)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.9)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def createData(npzDataDir):
     #load data from npz with small training data
    # DATAPATH = "../data/data_s.npz"
    # with np.load(DATAPATH) as data:
    #     features = data["feature"]
    #     labels = data["label"]
    # assert features.shape[0] == labels.shape[0]
    # dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    #load data from npz with large training data
    def getDataSet(DATAPATH,trainortest="train"):
        if os.path.isfile(DATAPATH) == False:
            raise FileNotFoundError("can not find npzfile")
        with np.load(DATAPATH) as data:
            feature = data["feature"]
            label = data["label"]
        assert feature.shape[0] == label.shape[0]
        return feature, label

    if os.path.isdir(npzDataDir):
        npzFileNames = os.listdir(npzDataDir)
        if len(npzFileNames) > 0:
            npzFilePATHs = list(map(lambda x: npzDataDir+"/"+x,npzFileNames))
            features,labels = getDataSet(npzFilePATHs[0])
            for index in range(1,len(npzFilePATHs)):
                feature,label = getDataSet(npzFilePATHs[index])
                np.concatenate(features,feature)
                np.concatenate(labels,label)
            else:
                assert features.shape[0] == labels.shape[0]
                # features_placeholder = tf.placeholder(features.dtype,features.shape)
                # labels_placeholder = tf.placeholder(labels.dtype,labels.shape)
                # dataset = tf.data.Dataset.from_tensor_slices(
                #     (features_placeholder, labels_placeholder)).shuffle(64).batch(32).prefetch(1)
                # iterator = dataset.make_initializable_iterator()
                # feature, label = iterator.get_next()
                # iterator_initializer_hook.iterator_initializer_func = \
                #     lambda sess: sess.run(
                #         iterator.initializer,
                #         feed_dict={features_placeholder: features,
                #                              labels_placeholder: labels})
                return features,labels

    else:
        raise Exception("cannot locate npzFileDir")


def main(unused_argv):
    #read data
    DATAPATH = "../data/policynetwork/data_policy_one_float32.npz"
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
    onehot_labels_t = tf.one_hot(indices=tf.cast(test_data_labels, tf.uint8), depth=138)
    print(test_data_labels)
    onehot_labels_tr = tf.Session().run(onehot_labels_t)
    print(onehot_labels_tr[2])

    #create estimator
    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="../model/policynetwork_eight_layer"
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
        batch_size=128,
        num_epochs=1,
        shuffle=True
    )

    cnn_classifier.train(
        input_fn=train_input_fn,
        hooks=[logging_hook])

    #evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_data_features},
        y=test_data_labels,
        num_epochs=1,
        shuffle=False
    )
    
    
    result = cnn_classifier.predict(
        input_fn=eval_input_fn
    )
    for x in result:
        print(x)
    
def newBiasAdd(output):
    shape = tf.shape(output)
    output_reshape = tf.reshape(output,[shape[0],shape[1]*shape[2],shape[3]])
    output_transpose = tf.transpose(output_reshape, [0, 2, 1])
    bias = tf.random_normal([64], mean=0, stddev=4,dtype=output.dtype)
    output_sum = tf.nn.bias_add(output_transpose,bias)
    output_back = tf.transpose(output_sum, [0, 2, 1])
    result = tf.reshape(output_back, [shape[0], shape[1],shape[2], shape[3]])
    return result
    
    
    # output_bias = 



if __name__ == "__main__":
    tf.app.run()

