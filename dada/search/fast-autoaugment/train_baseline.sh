#! /bin/sh
set -x
export PYTHONPATH=$PYTHONPATH:$PWD
GPUS=0,1,2,3
CONF=confs/wresnet40x2_cifar10_b512_test.yaml
CONF=confs/wresnet28x10_cifar10_b512_test.yaml
CONF=confs/shake26_2x112d_cifar_b512_test.yaml
CONF=confs/shake26_2x96d_cifar_b512_test.yaml
CONF=confs/wresnet28x10_cifar10_b512_test.yaml
CONF=confs/shake26_2x112d_cifar_b512_test.yaml
CONF=confs/shake26_2x96d_cifar_b512_test.yaml
CONF=confs/shake26_2x32d_cifar_b512_test.yaml
CONF=confs/wresnet28x10_svhn_b512_test.yaml
CONF=confs/wresnet40x2_cifar10_b512_test.yaml
CONF=confs/wresnet40x2_cifar10_b512.yaml
DATASET=svhn
DATASET=cifar10
DATASET=cifar100
GENOTYPE=k1_cifar10_search_epoch35_5
GENOTYPE=k1_cifar10_search_epoch200_batch512
GENOTYPE=k2_test_cifar10_search_epoch200_batch512
GENOTYPE=k2_test_cifar10_search_epoch200_batch512_5
GENOTYPE=k1_cifar10_test
GENOTYPE=k1_cifar10_search_epoch35
GENOTYPE=k2_test_cifar10_search_epoch37_batch512_10_50
GENOTYPE=k2_test_cifar10_search_epoch20_batch512_st_5
GENOTYPE=k2_test_cifar10_search_epoch20_batch512_st_10
GENOTYPE=k2_test_cifar10_search_epoch20_batch512_st_15
GENOTYPE=k2_test_cifar10_search_epoch20_batch512_stp_5
GENOTYPE=k2_test_cifar10_search_epoch20_batch512_stp_10
GENOTYPE=k2_test_cifar10_search_epoch20_batch512_stp_15
GENOTYPE=reduced_cifar100_wresnet40_2_128_100_10
GPUS=0,1,2,3,4,5,6,7
DATASET=svhn
CONF=confs/wresnet28x10_svhn_b1024.yaml
AUG=fa_reduced_cifar10
AUG=default
SAVE=weights/`basename ${CONF} .yaml`_${AUG}_${DATASET}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --aug ${AUG} --save ${SAVE}
# CONF=confs/wresnet28x10_svhn_b512.yaml
# CONF=confs/wresnet28x10_cifar100_b512.yaml
# CONF=confs/wresnet28x10_cifar100_b512_test_pba.yaml
# AUG=default
# CONF=confs/wresnet28x10_cifar100_b512_test.yaml
# AUG=default
# AUG=fa_reduced_cifar10
# SAVE=weights/`basename ${CONF} .yaml`_${AUG}_${DATASET}/test.pth
# CONF=confs/wresnet28x10_cifar100_b512_test_pba.yaml
# AUG=default
# CONF=confs/wresnet40x2_cifar100_b512_test.yaml
# AUG=fa_reduced_cifar10
# SAVE=weights/`basename ${CONF} .yaml`_${AUG}_${DATASET}/test.pth
# CONF=confs/wresnet28x10_cifar100_b512_test_pba.yaml
# AUG=fa_reduced_cifar10
# CONF=confs/resnet50_b512.yaml
# AUG=fa_reduced_imagenet
# DATASET=imagenet

