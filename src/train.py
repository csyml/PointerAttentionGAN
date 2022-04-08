from __future__ import absolute_import
from __future__ import division

import logging
from absl import app
import tensorflow as tf
from src.gan.documentation import setup_logdir, get_properties
from src.gan.documentation import print_run_metadata
from src.params.parameters import get_flags


def main(_args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    FLAGS = get_flags()
    properties=get_properties(FLAGS)
    logdir = setup_logdir(FLAGS, properties)
    print_run_metadata(FLAGS)
    noise=tf.random.truncated_normal([FLAGS.batch_size,FLAGS.seq_len],stddev=0.5,dtype=tf.float32,name='noise')
    # Trainning and summary in tensorboard
    pagan=PAGAN(Protein(flags,properties,logdir),noise))





if __name__=="__main__":
    app.run(main)