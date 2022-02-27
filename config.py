# coding=utf-8
import tensorflow as tf

flags = tf.app.flags

"""Data parameters"""
flags.DEFINE_string("run_id", "20220216_01", "")
flags.DEFINE_string("train_dir", "input/trian.tsv", "train data path")
flags.DEFINE_string("val_dir", "input/dev.tsv", "valid data path")
flags.DEFINE_string("ckpt_dir", "ckpt/", "checkpoint path")
flags.DEFINE_string("save_dir", "save/", "model saving path")
flags.DEFINE_string("vocab_dir", "vocab/vocab.pkL" , "vocabulary path")
flags.DEFINE_string("board_dir", "board/", "tensorboard summary dir")
flags.DEFINE_integer("seq_length" , 15, "max document length")
flags.DEFINE_integer("vocab_size", 7901, "vocabulary size")
flags.DEFINE_integer("num_classes", 2, "number of classes")
flags.DEFINE_integer("min_frequency", 0, "minimal word frequency")

"""Model hyperparameters"""
flags.DEFINE_integer("char_embedding_size", 128, "word embedding size")
flags.DEFINE_integer("hidden_size", 512, "hidden size in LSTM")
flags.DEFINE_integer("dense_output_size", 512, "output size of last dense Layer")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_float("keep_prob", 0.7, "dropout keep probality")
flags.DEFINE_float("initializer_range", 0.02, "The stdev of the truncated_normal_initializer for initiatizing all weight matrices")
flags.DEFINE_float("l2_lambda", 1e-4, "The weight of l2 reguLarization")
flags.DEFINE_float("warmup_proportion", 0.0, "Proportion of training to perform Linear Learning rate warmup for")

"""Training parameterss"""
flags.DEFINE_integer("epochs", 50, "number of epochs")
flags.DEFINE_integer("batch, size", 1024, "batch s1ze")

FLAGS = flags.FLAGS