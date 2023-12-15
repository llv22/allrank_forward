import os
from argparse import ArgumentParser, Namespace

import tensorflow as tf
from pathlib import Path
from sklearn.datasets import load_svmlight_file


def parse_args() -> Namespace:
    """
    parse_args - parses the arguments from the command line
    """
    parser = ArgumentParser("Normalize features script")
    parser.add_argument("--ds_path", help="location of the dataset", required=True, type=str)
    parser.add_argument("--out_tf_path", help="location of the output tf dataset", required=True, type=str)
    return parser.parse_args()

def serialize_example(features):
    feature = {}
    for index, f in enumerate(features):
        if index < len(features) - 1:
            feature[f"custom_features_{index+1}"] = tf.train.Feature(float_list=tf.train.FloatList(value=[f]))
        else:
            feature["utility"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[f]))
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

args = parse_args()
targets = ["train", "test", "vali"]

for target in targets:
    x, y, query_ids = load_svmlight_file(os.path.join(args.ds_path, f"{target}.txt"), query_id=True)
    folder = Path(args.out_tf_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    output_tfrecord_file = os.path.join(args.out_tf_path, f"{target}_numerical.tfrecord")
    with tf.io.TFRecordWriter(output_tfrecord_file) as writer:
        for i in range(x.shape[0]):
            features = x[i,:].toarray()[0].tolist()
            features.append((int)(y[i]))
            serialized_example = serialize_example(features)
            writer.write(serialized_example)


