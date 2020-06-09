import itertools
import os
import re
import time
from collections import defaultdict

import jieba
import numpy as np
import tensorflow as tf
import yaml

np.random.seed(3)

dataset_xiaohuangji = '/content/drive/My Drive/chatbot/seq.data'
dataset_chatterbot = '/content/drive/My Drive/chatbot/chatterbot_chinese/'
max_len = 16
max_vocab_size = 20000
units = 32
embedding_size = 128
init_lr = 1e-2
end_lr = 1e-5
batch_size = 64
epochs = 500

input_signature = [
    tf.TensorSpec((None, max_len), dtype=tf.int32),
    tf.TensorSpec((None, max_len), dtype=tf.int32)
]


class SimpleSeq2SeqModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, units):
        super(SimpleSeq2SeqModel, self).__init__()
        self.enc_units = units
        self.dec_units = units * 2
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        # encoder
        self.enc_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform'))
        # decoder
        self.dec_rnn = tf.keras.layers.GRU(
            self.dec_units, return_sequences=True, return_state=True)
        self.dec_fc = tf.keras.layers.Dense(vocab_size)
        # Bahdanau Attention
        self.att_w1 = tf.keras.layers.Dense(self.dec_units)
        self.att_w2 = tf.keras.layers.Dense(self.dec_units)
        self.att_v = tf.keras.layers.Dense(1)

    def bahdanau_attention(self, query, values):
        """Bahdanau Attention注意力机制
        query: shape=(batch_size, hidden_size)
        values: shape=(batch_size, seq_len, hidden_size)
        """
        query = tf.expand_dims(query, axis=1)
        # shape=(batch_size, seq_len, 1)
        score = self.att_v(tf.nn.tanh(self.att_w1(query) + self.att_w2(values)))
        weight = tf.nn.softmax(score, axis=1)
        # shape=(batch_size, hidden_size)
        context = tf.reduce_sum(weight * values, axis=1)
        return context, weight

    def encode(self, x, state=None):
        """编码"""
        if state is None:
            state = [tf.zeros((tf.shape(x)[0], self.enc_units)) for _ in range(2)]
        output, fw_state, bw_state = self.enc_rnn(self.embedding(x), initial_state=state)
        state = tf.concat((fw_state, bw_state), axis=1)
        return output, state

    def call(self, enc_outputs, dec_input, state):
        """序列到序列回调函数
        inputs: shape=(batch_size, inp_seq_len)
        target: shape=(batch_size, )
        state: shape=(batch_size, hidden_size)
        enc_output: shape=(batch_size, inp_seq_len, hidden_size)
        """
        embedding = self.embedding(dec_input)
        context, weight = self.bahdanau_attention(state, enc_outputs)
        # shape=(batch_size, 1, embedding_size + units)
        rnn_input = tf.expand_dims(tf.concat((context, embedding), axis=1), axis=1)
        output, state = self.dec_rnn(rnn_input, initial_state=state)
        # shape=(batch_size, vocab_size)
        logit = tf.reshape(self.dec_fc(output), shape=(-1, self.vocab_size))
        return logit, state, weight


class SimpleSeq2Seq(object):

    def __init__(self, vocab, embedding_size, units, init_lr, end_lr):
        self.vocab = vocab
        self.id2word, _ = zip(*sorted(self.vocab.items(), key=lambda x: x[1]))
        self.model = SimpleSeq2SeqModel(
            vocab_size=len(self.vocab), embedding_size=embedding_size, units=units)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_lr, decay_steps=1e10, end_learning_rate=end_lr)
        self.optimizer = tf.keras.optimizers.Adam(self.lr_schedule)

        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        self.metric_loss = tf.keras.metrics.Mean('mean_loss', dtype=tf.float32)

        self.time = 0

    def loss_function(self, real, pred):
        mask = tf.cast(tf.math.not_equal(real, 0), dtype=tf.float32)
        loss = self.loss_object(real, pred) * mask
        return tf.reduce_mean(loss)

    def train(self, train_dataset, steps):
        print('Total steps:%s' % steps)
        self.time = time.time()

        # reset learning rate decay steps
        self.lr_schedule.decay_steps = steps

        for inputs, targets in train_dataset:
            self.train_step(inputs, targets)

            if self.global_step % 100 == 0:
                self.report_metrics()

            if self.global_step.numpy() == steps:
                break

    def report_metrics(self):
        step = self.global_step.numpy()
        loss = self.metric_loss.result().numpy()
        self.metric_loss.reset_states()

        tm = time.time() - self.time
        print('seconds: %.2fs, step: %d, loss: %.4f' % (tm, step, loss))
        self.predict_batch(['什么 是 ai', '你 喝酒 吗', '蜘蛛侠 是 谁', '谁 是 蜘蛛侠', '喜欢 什么 编程 语言'])
        self.time = time.time()

    @tf.function(input_signature=input_signature)
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss = 0
            enc_output, state = self.model.encode(inputs)
            for t in range(1, targets.shape[1]):
                logit, state, _ = self.model(enc_output, targets[:, t - 1], state)
                loss += self.loss_function(targets[:, t], logit)
            loss = loss / tf.cast(tf.shape(inputs)[1], tf.float32)

        self.optimizer.apply_gradients(zip(tape.gradient(
            loss, self.model.trainable_variables), self.model.trainable_variables))
        self.global_step.assign_add(1)
        self.metric_loss(loss)

    def predict_batch(self, sentences, max_len=128):
        sent_ids = [Dataset.sent2ids(sent, self.vocab) for sent in sentences]
        sent_ids = Dataset.pad_sequence(sent_ids, None)

        # encoder output and initial state
        enc_output, state = self.model.encode(tf.convert_to_tensor(sent_ids))
        # initial decoder input
        target = tf.convert_to_tensor([self.vocab['<start>']] * len(sentences), dtype=tf.float32)
        pred_ids = []
        while len(pred_ids) < max_len:
            # logit, shape=(batch_size, 1, vocab_size)
            logit, state, _ = self.model(enc_output, target, state)
            target = tf.squeeze(tf.argmax(logit, axis=-1))
            pred_ids.append(target)

        predictions = []
        for ids in np.array(pred_ids, dtype=np.int32).T:
            sent = ''
            for idx in ids:
                if idx == self.vocab['<end>']:
                    break
                sent += self.id2word[idx]
            predictions.append(sent)

        for real, pred in zip(sentences, predictions):
            print('%s >> %s' % (real.replace(' ', ''), pred))


class Dataset(object):

    @classmethod
    def xiaohuangji(cls, corpus_path, max_len=16, max_vocab_size=2e4, batch_size=128):
        """读取小黄鸡语料"""
        dataset = []
        with open(corpus_path, encoding='utf8') as f:
            for line in f.readlines():
                if '=' in line:
                    continue
                try:
                    inp, tar = line.strip().split('\t')
                    inp = re.sub('\\s+', ' ', inp).strip()
                    tar = re.sub('\\s+', ' ', tar).strip()
                except ValueError:
                    continue
                dataset.append((inp, tar))
        return cls.build_dataset(dataset, max_len, max_vocab_size, batch_size)

    @classmethod
    def chatterbot(cls, corpus_path, max_len=16, max_vocab_size=20000, batch_size=32):
        """聊天机器人精分语料"""
        dataset = []
        for file in os.listdir(corpus_path):
            with open(os.path.join(corpus_path, file), encoding='utf8') as f:
                for dialogs in yaml.load(f)['conversations']:
                    dialogs = [' '.join(jieba.lcut(sentence)) for sentence in dialogs]
                    for i in range(1, len(dialogs)):
                        dataset.append((dialogs[i - 1], dialogs[i]))
        np.random.shuffle(dataset)

        return cls.build_dataset(dataset, max_len, max_vocab_size, batch_size)

    @classmethod
    def build_dataset(cls, dataset, max_len, max_vocab_size, batch_size):
        """Dataset Interface."""
        # build vocabulary
        counter = defaultdict(int)
        for sentence in itertools.chain(*dataset):
            for word in sentence.split(' '):
                counter[word] += 1
        words, _ = zip(*sorted(counter.items(), key=lambda x: x[1], reverse=True))
        vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, ' ': 3}
        vocab = dict(vocab, **dict(zip(words, range(len(vocab), max_vocab_size))))

        # sentence -> id
        inps, tars = [], []
        for inp, tar in dataset:
            inp = cls.sent2ids(inp, vocab)
            tar = cls.sent2ids(tar, vocab)
            if 1 <= len(inp) <= max_len and 1 <= len(tar) <= max_len:
                inps.append(inp)
                tars.append(tar)

        # padding sequence
        inps = cls.pad_sequence(inps, max_len)
        tars = cls.pad_sequence(tars, max_len)
        # using Dataset interface
        dataset = tf.data.Dataset.from_tensor_slices((inps, tars)) \
            .repeat() \
            .shuffle(10000) \
            .batch(batch_size, drop_remainder=True) \
            .prefetch(3)

        print('Total examples:%d, Vocab_size:%d' % (len(inps), len(vocab)))

        return dataset, vocab, len(inps)

    @staticmethod
    def sent2ids(sent, vocab):
        """转换句子为id"""
        sent = '<start> ' + re.sub('\\s', ' ', sent).strip() + ' <end>'
        ids = []
        for word in sent.split(' '):
            if not word:
                word = ' '
            if word in vocab:
                ids.append(vocab[word])
        return ids

    @staticmethod
    def pad_sequence(sequence, max_len=None):
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0)


dataset, vocab, n_examples = Dataset.chatterbot(
    dataset_chatterbot, max_len=max_len, max_vocab_size=max_vocab_size, batch_size=batch_size)

model = SimpleSeq2Seq(
    vocab=vocab,
    embedding_size=embedding_size,
    units=units,
    init_lr=init_lr,
    end_lr=end_lr)
steps = epochs * (n_examples // batch_size)
model.train(train_dataset=dataset, steps=steps)
