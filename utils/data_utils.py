# coding=utf-8
import os
import re
import numpy as np
import csv
from tensorflow.contrib import learn
import time
import config

FLAGS = config.FLAGS

def load_data(file_path, num_classes, max_length, min_frequency=0, vocab_processor=None, vocab_path=None):
    """ 加载数据，读取tsv文件 """
    start = time.time()
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        querys = []
        docs = []
        labels = []
        for line in reader:
            query = line[0].strip().lower()
            doc = line[1].strip().lower()
            if len(query) < 1 and len(doc) < 1:
                print("filter len < 1", query, doc)
                continue
            query = _word_segmentation(query, max_length)
            doc = _word_segmentation(doc, max_length)
            querys.append(query)
            docs.append(doc)
            label = list(np.zeros(num_classes, int))
            label[int(line[2])] = 1
            labels.append(label)

    # Real length
    query_lens = np.array(list(map(len, [sent.split(' ') for sent in querys])))
    doc_lens = np.array(list(map(len, [sent.split(' ') for sent in docs])))
    real_max_doc_length = max(max(query_lens), max(doc_lens))
    if max_length == 0:
        max_length = real_max_doc_length
    elif real_max_doc_length > max_length:
        query_lens = np.array([max_length if l > max_length else l for l in query_lens])
        doc_lens = np.array([max_length if l > max_length else l for l in doc_lens])

    # Extract vocabulary from sentences and map words to indices
    if vocab_processor is None:
        vocab = None
        if vocab_path != None and len(vocab_path) != 0:
            vocab = learn.preprocessing.CategoricalVocabulary()
            with open(vocab_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    vocab.add(line)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length,
                                                                  min_frequency=min_frequency,
                                                                  vocabulary=vocab)
        query_data = np.array(list(vocab_processor.fit_transform(querys)))
        doc_data = np.array(list(vocab_processor.fit_transform(docs)))
    else:
        query_data = np.array(list(vocab_processor.transform(querys)))
        doc_data = np.array(list(vocab_processor.transform(docs)))

    query_data_size = len(query_data)
    doc_data_size = len(doc_data)
    if query_data_size != doc_data_size:
        print("Data size not equal", query_data_size, doc_data_size)

    end = time.time()

    print('-' * 10, + "query" + '-' * 10)
    print(querys[:10])
    print(query_data[:10, :])
    print('-' * 10, + "doc" + '-' * 10)
    print(docs[:10])
    print(doc_data[:10, :])

    print("\nDataset has been built seccessfully.")
    print("Run time: {}".format(end-start))
    print("Number of sentences: {}".format(query_data_size))
    print("Vocabulary size: {}".format(len(vocab_processor.vocabulary_._mapping)))
    print("Max document length: {}".format(vocab_processor.max_document_length))
    print("query data shape: {}".format(query_data.shape))
    print("query lengths shape: {}\n".format(query_lens.shape))

    return (query_data, doc_data), labels, (query_lens, doc_lens), vocab_processor, (querys, docs)


def _word_segmentation(sent, max_length=0):
    """ Tokenizer for Chinese """
    p = re.compile(r"([\u4e00-\u9fa5]{1}|[a-z]{1,}|[0-9]{1,})")
    match_group = p.findall(sent)
    if max_length != 0:
        match_group = match_group[0: max_length]
    sent = ' '.join(match_group)
    return re.sub(r"\s+", ' ', sent)


def data_iterator(query, doc, label, query_len, doc_len):
    dataset = tf.data.Dataset.from_tensor_slices((query, doc, label, query_len, doc_len))
    dataset = dataset.batch(FLAGS.batch_size).repeat(FLAGS.epochs)
    iterator = dataset.make_initializable_iterator()
    next_element = itertools.get_next()
    return iterator, next_element