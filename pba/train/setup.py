"""Parse flags and set up hyperparameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from augmentation_transforms_hp import NUM_HP_TRANSFORM


def create_parser(state):
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, 
        default='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/tiger/patches/',
        help='Directory where dataset is located.'
        )
    parser.add_argument(
        '--model_name',
        default='resnet',
        choices=('wrn_28_10', 'wrn_40_2', 'shake_shake_32', 'shake_shake_96',
                 'shake_shake_112', 'pyramid_net', 'resnet'))
    parser.add_argument(
        '--dataset',
        default='tiger',
        choices=('camelyon17','tiger', 'cifar100', 'svhn', 'svhn-full', 'test'))
    parser.add_argument(
        '--recompute_dset_stats',
        action='store_true',
        help='Instead of using hardcoded mean/std, recompute from dataset.')
    parser.add_argument(
        '--local_dir', 
        type=str, 
        default='/data/pathology/projects/autoaugmentation/from_chansey_review/pba_colab/tmp/ray_results/',
        help='Ray directory.')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Checkpoint frequency.')
    parser.add_argument(
        '--cpu', type=float, default=1, help='Allocated by Ray')
    parser.add_argument(
        '--gpu', type=float, default=1, help='Allocated by Ray')
    parser.add_argument(
        '--aug_policy',
        type=str,
        default='cifar10',
        help='which augmentation policy to use (in augmentation_transforms_hp.py)')
    # search-use only
    parser.add_argument(
        '--explore',
        type=str,
        default='camelyon17',
        help='which explore function to use')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs, or <=0 for default')
    parser.add_argument(
        '--no_cutout', action='store_true', help='turn off cutout')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.005, help='weight decay')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--test_bs', type=int, default=32, help='test batch size')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of Ray samples')

    if state == 'train':
        parser.add_argument('--restore', type=str, default='') #val-val_cwh_lpe_-ts-109015-bs-64-nwd-adam-train/autoaug/RayModel_0_2022-01-08_13-46-45m9z0n4fq/checkpoint_1/model.ckpt-1   

        parser.add_argument(
            '--val_set',
            default='val_A2_jb_',
            choices=('val_rh_umcu_', 'val_cwh_umcu_', 'val_rh_lpe_', 'val_cwh_lpe_','val_A2_jb_','val_E2_BH_'))
        parser.add_argument(
            '--train_set', type=str,
            default='.npy',
            choices=('_25000.npy', '.npy'))
        parser.add_argument(
            '--ind', type=str,
            default='0')
        parser.add_argument(
            '--train_size', 
            type=int, 
            default=536121, 
            help='Number of training examples.')  # 109015 #25000 #536121
        parser.add_argument(
            '--val_size', 
            type=int, 
            default=3600, 
            help='Number of validation examples.')
        parser.add_argument(
            '--use_hp_policy',
            type=bool, 
            default=True, 
            help='otherwise use autoaug policy')
        parser.add_argument(
            '--hp_policy',
            type=str,
            default='/data/pathology/projects/autoaugmentation/from_chansey_review/pba_colab/tmp/ray_results/val-cwh_lpe_-ts-25000-bs-64-pi-3-ns-16/autoaug_pbt/pbt_policy_8.txt',
            help='either a comma separated list of values or a file')
        parser.add_argument(
            '--hp_policy_epochs',
            type=int,
            default=100,
            help='number of epochs/iterations policy trained for')
        parser.add_argument(
            '--no_aug',
            action='store_true',
            help=
            'no additional augmentation at all (besides cutout if not toggled)'
            )
        parser.add_argument(
            '--flatten',
            action='store_true',
            help='randomly select aug policy from schedule')
        parser.add_argument('--name', type=str, default='autoaug')

    elif state == 'search':
        parser.add_argument(
            '--perturbation_interval', 
            type=int, 
            default=3) # 10
        parser.add_argument(
            '--name', 
            type=str, 
            default='autoaug_pbt')
        parser.add_argument(
            '--val_set',
            default='val_cwh_lpe_',
            choices=('val_rh_umcu_', 'val_cwh_umcu_', 'val_rh_lpe_', 'val_cwh_lpe_'))
        parser.add_argument(
            '--train_set', type=str,
            default='_25000.npy',
            choices=('_25000.npy', '.npy'))
        parser.add_argument(
            '--train_size', 
            type=int, 
            default=25000, 
            help='Number of training examples.') # 109015 # 25000
        parser.add_argument(
            '--val_size', 
            type=int, 
            default=3600, 
            help='Number of validation examples.')
    elif state == 'infer':
        parser.add_argument(
            '--perturbation_interval', 
            type=int, 
            default=3) # 10
        parser.add_argument(
            '--name', 
            type=str, 
            default='autoaug_pbt')
        parser.add_argument(
            '--val_set',
            default='test_umcu_',
            choices=('test_umcu_', 'test_cwh_', 'test_rh_', 'test_lpe_'))
        parser.add_argument(
            '--train_set', type=str,
            default='_25000.npy',
            choices=('_25000.npy', '.npy'))

        parser.add_argument(
            '--infer_path', type=str,
            default='val-val_cwh_lpe_-ts-109015-bs-64-nwd-adam-train/autoaug/RayModel_0_2022-01-08_13-46-45m9z0n4fq/checkpoint_100/model.ckpt-100',
            help='Path of the inference model')
        parser.add_argument(
            '--train_size', 
            type=int, 
            default=25000, 
            help='Number of training examples.') # 109015 # 25000
        parser.add_argument(
            '--val_size', 
            type=int, 
            default=3600, 
            help='Number of validation examples.')
        
    else:
        raise ValueError('unknown state')
    args = parser.parse_args()
    if state == 'train':
        args.local_dir = "{dir}val-{vs}-ts-{ts}-bs-{bs}-nwd-adam-train-{ind}-".format(
        dir =args.local_dir,vs =args.val_set,ts =args.train_size,
        bs =args.bs,ind =args.ind)

    elif state == 'search':
        args.local_dir = "{dir}val-{vs}-ts-{ts}-bs-{bs}-pi-{pi}-ns-{ns}-nwd-adam".format(
        dir =args.local_dir,vs =args.val_set,ts =args.train_size,
        bs =args.bs,pi =args.perturbation_interval,ns =args.num_samples)
    elif state == 'infer':
        args.restore = args.local_dir + args.infer_path
        #'val-cwh_lpe_-ts-109015-bs-64-nwd-adam-train/autoaug/RayModel_0_2021-12-29_21-54-362_6ppmj3/checkpoint_100/model.ckpt-100'
        #
    tf.logging.info(str(args))
    return args


def create_hparams(state, FLAGS):  # pylint: disable=invalid-name
    """Creates hyperparameters to pass into Ray config.

    Different options depending on search or eval mode.

    Args:
        state: a string, 'train' or 'search'.
        FLAGS: parsed command line flags.

    Returns:
        tf.hparams object.
    """
    epochs = 0
    tf.logging.info('data path: {}'.format(FLAGS.data_dir))
    hparams = tf.contrib.training.HParams(
        train_size=FLAGS.train_size,
        local_dir=FLAGS.local_dir,
        validation_size=FLAGS.val_size,
        train_set=FLAGS.train_set,
        val_set=FLAGS.val_set,
        dataset=FLAGS.dataset,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.bs,
        gradient_clipping_by_global_norm=5.0,
        explore=FLAGS.explore,
        aug_policy=FLAGS.aug_policy,
        no_cutout=FLAGS.no_cutout,
        recompute_dset_stats=FLAGS.recompute_dset_stats,
        lr=FLAGS.lr,
        weight_decay_rate=FLAGS.wd,
        test_batch_size=FLAGS.test_bs)

    if state == 'train':
        hparams.add_hparam('no_aug', FLAGS.no_aug)
        hparams.add_hparam('use_hp_policy', FLAGS.use_hp_policy)
        hparams.add_hparam('ind', FLAGS.ind)
    
        if FLAGS.use_hp_policy:
            if FLAGS.hp_policy == 'random':
                tf.logging.info('RANDOM SEARCH')
                parsed_policy = []
                for i in range(NUM_HP_TRANSFORM * 4):
                    if i % 2 == 0:
                        parsed_policy.append(random.randint(0, 10))
                    else:
                        parsed_policy.append(random.randint(0, 9))
            elif FLAGS.hp_policy.endswith('.txt') or FLAGS.hp_policy.endswith(
                    '.p'):
                # will be loaded in in data_utils
                parsed_policy = FLAGS.hp_policy
            else:
                # parse input into a fixed augmentation policy
                parsed_policy = FLAGS.hp_policy.split(', ')
                parsed_policy = [int(p) for p in parsed_policy]
            hparams.add_hparam('hp_policy', parsed_policy)
            hparams.add_hparam('hp_policy_epochs', FLAGS.hp_policy_epochs)
            hparams.add_hparam('flatten', FLAGS.flatten)
    elif state == 'search':
        hparams.add_hparam('no_aug', False)
        hparams.add_hparam('use_hp_policy', True)
        # default start value of 0
        hparams.add_hparam('hp_policy',
                           [0 for _ in range(4 * NUM_HP_TRANSFORM)])
    elif state == 'infer':
        hparams.add_hparam('no_aug', False)
        hparams.add_hparam('use_hp_policy', True)
        # default start value of 0
        hparams.add_hparam('hp_policy',
                           [0 for _ in range(4 * NUM_HP_TRANSFORM)])
    else:
        raise ValueError('unknown state')

    if FLAGS.model_name == 'wrn_40_2':
        hparams.add_hparam('model_name', 'wrn')
        epochs = 200
        hparams.add_hparam('wrn_size', 32)
        hparams.add_hparam('wrn_depth', 40)
    elif FLAGS.model_name == 'wrn_28_10':
        hparams.add_hparam('model_name', 'wrn')
        epochs = 200
        hparams.add_hparam('wrn_size', 160)
        hparams.add_hparam('wrn_depth', 28)
    elif FLAGS.model_name == 'resnet':
        hparams.add_hparam('model_name', 'resnet')
        epochs = 100
        hparams.add_hparam('resnet_size', 20)
        hparams.add_hparam('num_filters', 64)
    elif FLAGS.model_name == 'shake_shake_32':
        hparams.add_hparam('model_name', 'shake_shake')
        epochs = 1800
        hparams.add_hparam('shake_shake_widen_factor', 2)
    elif FLAGS.model_name == 'shake_shake_96':
        hparams.add_hparam('model_name', 'shake_shake')
        epochs = 1800
        hparams.add_hparam('shake_shake_widen_factor', 6)
    elif FLAGS.model_name == 'shake_shake_112':
        hparams.add_hparam('model_name', 'shake_shake')
        epochs = 1800
        hparams.add_hparam('shake_shake_widen_factor', 7)
    elif FLAGS.model_name == 'pyramid_net':
        hparams.add_hparam('model_name', 'pyramid_net')
        epochs = 1800
        hparams.set_hparam('batch_size', 64)
    else:
        raise ValueError('Not Valid Model Name: %s' % FLAGS.model_name)
    if FLAGS.epochs > 0:
        tf.logging.info('overwriting with custom epochs')
        epochs = FLAGS.epochs
    hparams.add_hparam('num_epochs', epochs)
    tf.logging.info('epochs: {}, lr: {}, wd: {}'.format(
        hparams.num_epochs, hparams.lr, hparams.weight_decay_rate))
    return hparams
