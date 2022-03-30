import os
import json
import tensorflow as tf
import logging


from termcolor import colored

def setup_logdir(flags, properties):
    one_hot = "one_hot" if flags.one_hot else "embedding"
    input_size = "{}x{}_{}".format(flags.embedding_height, properties['seq_length'], one_hot)

    logdir = os.path.join(flags.weights_dir, flags.dataset.replace("\\", os.sep), flags.model_type, flags.architecture,
                          input_size, flags.loss_type)
    tf.io.gfile.mkdir(logdir)
    logging.info("Results will be saved in:{}".format(logdir.replace("\\", "\\\\")))
    return logdir


def get_properties(flags):
    path = os.path.join(flags.logdir, flags.properties)
    with open(path) as json_file:
        properties = json.load(json_file)
    return properties


def print_run_metadata(flags):
    print("Running model {} with {} data set.".format(colored(flags.model_type, 'red', attrs=['bold']),
                                                      colored(flags.dataset, 'red', attrs=['bold'])))
    print("Batch size:{}".format(colored(flags.batch_size, attrs=['bold'])))
    print("Learning rates used: discriminator {} ,generator {} (Beta:{}).".format(
        colored(flags.discriminator_learning_rate, attrs=['bold']),
        colored(flags.generator_learning_rate, attrs=['bold']),
        colored(flags.beta1, attrs=['bold'])
    ))
    print("Generator {} dim. Discriminator {} dim.".format(colored(flags.gf_dim, 'red', attrs=['bold']),
                                                           colored(flags.df_dim, 'red', attrs=['bold'])))
    print("G_step: {}, D_step:{}.".format(colored(flags.d_step, 'red', attrs=['bold']),
                                          colored(flags.g_step, 'red', attrs=['bold'])))
    print("\n")
    print(colored('!!! STARTING TRAINING!!!', attrs=['bold']))



