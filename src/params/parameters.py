import os
from absl import flags

CURRENT_DIRECTORY = os.path.dirname(__file__)
MODEL_TYPE="PointerAttentionGAN"
#parameters of  training
flags.DEFINE_string('dataset','dataset','Dataset to use for training. ')
flags.DEFINE_string('properties', os.path.join(CURRENT_DIRECTORY,'properties.json'), 'properties file. [properties.json]')
flags.DEFINE_integer('batch_size',16,'Number of protein sequences in input batch.[64]')
flags.DEFINE_boolean('is_train', True, 'True for training. [default: True]')
flags.DEFINE_integer('kernel_height', 3, 'The height of the kernel [3]')
flags.DEFINE_integer('kernel_width', 3, 'The width of the kernel [3]')
flags.DEFINE_integer('save_summary_steps', 10, 'Number of steps between saving summary statistics [300]')
flags.DEFINE_integer('save_checkpoint_sec', 100, 'Number of seconds between saving checkpoints of model [1200(20min)]')
flags.DEFINE_string('weights_dir', os.path.join(CURRENT_DIRECTORY, 'log'.replace("\\",os.sep), 'weights'.replace("\\", os.sep)),
                    'Location where all weights should be saved')
flags.DEFINE_string('logdir', os.path.join(CURRENT_DIRECTORY, 'log'.replace("\\", os.sep)),
                    'Location where all log is stored')

#parameters of  gan
flags.DEFINE_string('model_type', MODEL_TYPE, 'Model used for training. [model]')
flags.DEFINE_float('discriminator_learning_rate', 0.0001, 'Learning rate of for adam. [0.0004]')
flags.DEFINE_float('generator_learning_rate', 0.0001, 'Learning rate of for adam. [0.0004]')
flags.DEFINE_float('beta1', 0.0, 'Momentum term of adam. [0.5]')
flags.DEFINE_float('beta2', 0.9, 'Second momentum term of adam. [0.999]')
flags.DEFINE_integer('z_dim', 512, 'Dimensionality of latent code z. [8192]')
flags.DEFINE_integer('gf_dim', 48, 'Dimensionality of gf. [64]')
flags.DEFINE_integer('df_dim', 36, 'Dimensionality of df. [64]')
flags.DEFINE_string('loss_type', 'hinge_loss',
                    'the loss type can be [sngan-gp, hinge_loss, hinge_loss_ra, wasserstein or kl_loss, ipot, non_saturating]')
flags.DEFINE_integer('d_step', 1, 'The number of D_step')
flags.DEFINE_integer('g_step', 1, 'The number of G_step')

# parameters of protein
flags.DEFINE_integer('seq_len',512,'Length of protein sequence[512]')
flags.DEFINE_integer('steps_for_blast', 1200, 'Number of steps between blasting fake protein [1200]')
flags.DEFINE_bool('one_hot', True, 'Whether to use one hot encoding [False]')
flags.DEFINE_integer('embedding_height', 58, 'The height of embedding used in generator/discriminator')
flags.DEFINE_string('embedding_name', 'prot_fp', 'The height of embedding used in generator/discriminator')
flags.DEFINE_string('pooling', 'avg', 'Pooling [avg, conv, subpixel, None]')
flags.DEFINE_integer('dilation_rate', 2, 'The rate of the dilation [2]')
flags.DEFINE_float('noise_level', 1.0, 'Level of noise which is added to real data. [1.0]')
flags.DEFINE_float('variation_level', 0.0, 'Hyper parameter of loss which controls variation. [10]')
flags.DEFINE_float('label_noise_level', 0.02, 'Hyper parameter of noise level for label swapping. [0.05]')
flags.DEFINE_string('embeding_path', 'dataset', 'Relative location of embeddings for amino acids')
flags.DEFINE_integer('compound_w', 512, 'The maximum length of compound written in SMILES format [128]')
flags.DEFINE_integer('smiles_embedding_h', 512, 'The size of SMILES embedding [8] ')
flags.DEFINE_bool('static_embedding', True, 'Whether to use static pre-computed embeddings [True]')
flags.DEFINE_bool('dynamic_padding', True, 'Whether to use dynamic padding [False]')
flags.DEFINE_bool('already_embedded', False, 'Whether to use dynamic padding [False]')
flags.DEFINE_string('variable_path', '/bert/weights/tpu_fine_tuned/model.ckpt-1030000',
                    'Path of the checkpoint file where weights are stored for network part that '
                    'converts to embeddings to ids')
flags.DEFINE_string('blast_db', 'db', 'Location where fasta db is stored')


def get_flags():
    return flags.FLAGS