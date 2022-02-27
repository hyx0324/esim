# coding=utf-8
import os
import sys
from model.esim import Graph
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.saved_model import tag_constants
from utils.data_utils import load_data, data_iterator
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

FLAGS = config.FLAGS

# Load train data
text_data, labels, lengths, vocab_processor, sentences = load_data(file_path=FLAGS.train_dir,
                                                                   num_classes=FLAGS.num_classes,
                                                                   max_length=FLAGS.seq_length,
                                                                   min_frequency=FLAGS.min_frequency)
# Save vocabulary processor
vocab_processor.save(FLAGS.vocab_dir)
FLAGS.vocab_size = len(vocab_processor.vocabulary_._mapping)

# Load valid data
text_data_eval, labels_eval, lengths_eval, _, sentences_eval = load_data(file_path=FLAGS.val_dir,
                                                                         num_classes=FLAGS.num_classes,
                                                                         max_length=FLAGS.seq_length,
                                                                         min_frequency=FLAGS.min_frequency,
                                                                         vocab_processor=vocab_processor)

train_iterator, train_data_next = data_iterator(text_data[0], text_data[1], labels, lengths[0], lengths[1])
eval_iterator, eval_data_next = data_iterator(text_data_eval[0], text_data_eval[1], labels_eval, lengths_eval[0], lengths_eval[1])

steps = int(len(labels) / FLAGS.batch_size)
eval_steps = int(len(labels_eval) / FLAGS.batch_size)

model = Graph()
saver.tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_grouth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.board_dir, FLAGS.run_id, "train"), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join(FLAGS.board_dir, FLAGS.run_id, "eval"))
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    sess.run(eval_iterator.initializer)
    print('*' * 10 + " Start Training " + '*' * 10)
    for epoch in range(FLAGS.epochs):
        loss = 0.0
        loss_eval = 0.0
        # 初始化metrics的局部变量
        sess.run(tf.local_variables_initializer())
        for step in range(steps):
            query_batch, doc_batch, label_batch, query_len_batch, doc_len_batch = sess.run(train_data_next)
            _, cur_loss, acc, prec, recall, summary = sess.run(
                [model.train_op, model, loss, model.metrics["accuracy"][1],
                 model.metrics["precision"][1], model.metrics["recall"][1],
                 model.merged],
                feed_dict={model.p: query_batch,
                           model.h: doc_batch,
                           model.y: label_batch,
                           model.p_seq_len: query_len_batch,
                           model.h_seq_len: doc_len_batch,
                           model.keep_prob: FLAGS.keep_prob})
            loss += cur_loss
            if step != 0 and step % 500 == 0:
                print("epoch: %d, step: %d, train_loss: %-4.3f, train_acc: %-4.3f, train_pred: %-4.3f, train_recall: %-4.3f"
                      % (epoch, step, loss/(step+1), acc, prec, recall))

        train_writer.add_summary(summary, epoch)
        print("epoch: %d, step: %d, train_loss: %-4.3f, train_acc: %-4.3f, train_pred: %-4.3f, train_recall: %-4.3f"
              % (epoch, step, loss / steps, acc, prec, recall))
        sess.run(tf.local_variables_initializer())
        print('*' * 10 + " Start Eval " + '*' * 10)
        for eval_step in range(eval_steps):
            query_batch_eval, doc_batch_eval, label_batch_eval, query_len_batch_eval, doc_len_batch_eval = sess.run(eval_data_next)
            cur_loss_eval, acc_eval, prec_eval, recall_eval, summary_eval = sess.run(
                [model.train_op, model, loss, model.metrics["accuracy"][1],model.metrics["precision"][1],
                 model.metrics["recall"][1],model.merged],
                feed_dict={model.p: query_batch,
                           model.h: doc_batch,
                           model.y: label_batch,
                           model.p_seq_len: query_len_batch,
                           model.h_seq_len: doc_len_batch,
                           model.keep_prob: 1})
            loss_eval += cur_loss_eval
        eval_writer.add_summary(summary_eval, epoch)
        print("epoch: %d, step: %d, eval_loss: %-4.3f, eval_acc: %-4.3f, eval_pred: %-4.3f, eval_recall: %-4.3f"
              % (epoch, step, loss_eval / steps, acc_eval, prec_eval, recall_eval))

        builder = tf.save_model.builder.SavedModelBuilder(os.path.join(FLAGS.save_dir, FLAGS.run_id))
        signature = predict_signature_def(inputs={"query_input": model.p,
                                                  "doc_input": model.h,
                                                  "query_len": model.p_seq_len,
                                                  "doc_len": model.h_seq_len,
                                                  "keep_prob": model.keep_prob},
                                          outputs={"output": model.prob})
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={"predict": signature})
        builder.save()
        print("*" * 10 + " Model Save Successfully " + '*' * 10)

















