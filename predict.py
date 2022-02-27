# coding=utf-8
import os
import tensorflow
from tensorflow.saved_model import tag_constants
from tensorflow.contrib import learn
from utils.data_utils import load_data
import argparse
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

model_path_version = {
    "v1": "save/20220217_01"
}

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", help="input data path")
parser.add_argument("--output_path", help="output data path")
parser.add_argument("--version", help="model version")
parser.add_argument("--vocab_path", default="vocab/vocab.pkl", help="vocabulary path")
parser.add_argument("--num_calsses", default=2, help="number of classes")
parser.add_argument("--seq_length", default=15, help="max document length")
parser.add_argument("--min_frequency", default=0, help="minimal word frequency")
args = parser.parse_args()
print(args)

# Restore vocabulary process
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(args.vocab_path)
text_data, labels, lengths, vocab_processor, sentences = load_data(file_path=args.input_path,
                                                                   num_classes=args.num_classes,
                                                                   max_length=args.seq_length,
                                                                   min_frequency=args.min_frequency,
                                                                   vocab_processor=vocab_processor)
model_path = model_path_version[args.version]

with tf.session() as sess:
    model = tf.saved_model.loader.load(export_dir=model_path, sess=sess, tags=[tag_constants.SERVING])
    graph = tf.get_default_graph()
    query_input = graph.get_tensor_by_name("premise:0")
    doc_input = graph.get_tensor_by_name("hypothesis:0")
    query_len = graph.get_tensor_by_name("premise_seq_len:0")
    doc_len = graph.get_tensor_by_name("hypothesis_seq_len:0")
    keep_prob = graph.get_tensor_by_name("drop_rate:0")
    predict_probs = graph.get_tensor_by_name("Softmax:0")
    output = sess.run(predict_probs, feed_dict={query_input: text_data[0],
                                                doc_input: text_data[1],
                                                query_len: lengths[0],
                                                doc_len: lengths[1],
                                                keep_prob:1})


output = output[:, 1:]

test_data = pd.read_csv(args.input_path, sep="\t", names=["query", "doc", "label", "score"])
test_data["new_score"] = output
print(test_data.head())

test_data.to_csv(args.output_path + '-' + args.version + ".tsv", sep="\t", index=False)
print('*' * 10 + " End Predict " + '*' * 10)