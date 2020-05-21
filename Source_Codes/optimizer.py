

import sys
import argparse

from keras.models import load_model 
new_model =load_model('pro_dl.h5')
#new_model.get_weights()
new_model.optimizer

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model


# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = './model'
model_fname = 'pro_dl.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='pro_dl_frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model(model_fname)

session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)


import platform
is_win = 'windows' in platform.platform().lower()

# OpenVINO 2019
if is_win:
    mo_tf_path = '"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo_tf.py"'
else:
    # mo_tf.py path in Linux
    mo_tf_path = '/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py'

pb_file = './model/pro_dl_frozen_model.pb'
output_dir = './model'
img_height = 128
input_shape = [1,img_height,img_height,3]
input_shape_str = str(input_shape).replace(' ','')
input_shape_str
#!python {mo_tf_path} --input_model {pb_file} --output_dir {output_dir} --input_shape {input_shape_str} --data_type FP32


