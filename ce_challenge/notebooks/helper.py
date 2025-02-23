import tensorflow as tf
import time

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def exec_time_and_y_pred(model, dataset):
    # Measure the time taken for prediction
    start_time = time.time()
    y_pred = model.predict(dataset)
    end_time = time.time()

    # Calculate the total time taken for prediction
    total_time = end_time - start_time

    # Calculate the number of samples in the test data
    num_samples = len(dataset)

    # Calculate the execution time per sample
    time_per_sample = total_time / num_samples
    return total_time, time_per_sample, y_pred

# Function to calculate FLOPS
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    #concrete_func = concrete.get_concrete_function(tf.TensorSpec([1] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    concrete_func = concrete.get_concrete_function(tf.TensorSpec([1, *model.inputs[0].shape[1:]], model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_func.graph.as_graph_def()

    # Calculate FLOPS
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                          run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops