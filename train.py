# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from hparams import Hparams
import math
# import logging
from utils import logger

# logging.basicConfig(level=logging.INFO)


logger.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logger.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.batch_size,
                                             shuffle=True)
# logger.info(train_batches)
# logger.info(num_train_batches)
# logger.info(num_train_samples)

eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2,
                                             100000, 100000,
                                             hp.vocab, hp.batch_size,
                                             shuffle=False)

# create a iterator of the correct shape and type
# create
logger.info(train_batches.output_types)
logger.info(train_batches.output_shapes)
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()   # 、？？  # 通过session.run()  获得一个batch
logger.info(xs)
logger.info(ys)
# initial
train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)
logger.info(train_init_op)
logger.info(xs)
logger.info(ys)

logger.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(xs, ys)  # 输入数据和训练结合， train_op 为optimizer
y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logger.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logger.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    # create a writer
    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op) # 将数据加载到dataIter

    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps+1)):
        # session.run 可以直接获取batchdata 以及 loss
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logger.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss

            logger.info("# test evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logger.info("# get hypotheses")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            logger.info("# write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            logger.info("# calc bleu score and append it to translation")
            calc_bleu(hp.eval3, translation)

            logger.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logger.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logger.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logger.info("Done")
