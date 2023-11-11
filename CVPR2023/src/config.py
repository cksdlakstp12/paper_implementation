import os

from easydict import EasyDict as edict

import torch
import numpy as np

from utils.transforms import RandomErasing, ComposeForST, RandomHorizontalFlipForST
from utils.transforms import *

dataset_type = "KAIST" # KAIST or CVC14

# Dataset path
PATH = edict()

if dataset_type == "KAIST":
    PATH.DB_ROOT = '../../data/kaist-rgbt/'
    PATH.JSON_GT_FILE = os.path.join('kaist_annotations_test20.json' )
elif dataset_type == "CVC14":
    PATH.DB_ROOT = '../data/CVC-14/'
    PATH.JSON_GT_FILE = os.path.join('CVC14_annotations_test.json' )

# Soft Teacher
soft_teacher = edict()
if dataset_type == "KAIST":
    soft_teacher.student_checkpoint = "../KAIST_teacher_weights.pth.tar071"
    soft_teacher.teacher_checkpoint = "../KAIST_teacher_weights.pth.tar071"
elif dataset_type == "CVC14":
    soft_teacher.student_checkpoint = "../CVC_teacher_weights.pth.tar071"
    soft_teacher.teacher_checkpoint = "../CVC_teacher_weights.pth.tar071"

# KMeans
# kmeans = edict()
# kmeans.model_path = "kmeans_4chVector_model.pkl"
# kmeans.dict_path = "4chVector_mean_std_dict.pkl"

# train
train = edict()

train.three_way = True

train.soft_update_mode = "batch" # epoch or batch

train.day = "all"

if dataset_type == "KAIST":
    train.img_set = f"Labeled_Unlabled_combine.txt"
    train.teacher_img_set = "Unlabeled_90.txt"
    train.U_img_set = "Unlabeled_90.txt"
    train.L_img_set = "Labeled_10.txt"
    # train.teacher_img_set = "u_fast_test.txt"
    # train.U_img_set = "u_fast_test.txt"
    # train.L_img_set = "l_fast_test.txt"
elif dataset_type == "CVC14":
    train.img_set = f"CVC_100.txt"
    train.teacher_img_set = "CVC_Unlabeled_90.txt"
    train.U_img_set = "CVC_Unlabeled_90.txt"
    train.L_img_set = "CVC_Labeled_10.txt"

train.checkpoint = None ## Load chekpoint

train.batch_size = 6 # batch size

train.start_epoch = 0  # start at this epoch
train.epochs = 80  # number of epochs to run without early-stopping
train.epochs_since_improvement = 3  # number of epochs since there was an improvement in the validation metric
train.best_loss = 100.  # assume a high loss at first

train.lr = 1e-4   # learning rate
train.momentum = 0.9  # momentum
train.weight_decay = 5e-4  # weight decay
train.grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

train.min_score = 0.5

train.print_freq = 10

train.annotation = "AR-CNN" # AR-CNN, Sanitize, Original 

train.random_seed = 42

train.aug_swap_epoch = 0

# test & eval
test = edict()

test.result_path = '../result'
### coco tool. Save Results(jpg & json) Path

test.day = "all" # all, day, night
if dataset_type == "KAIST":
    test.img_set = f"test-{test.day}-20.txt"
    # test.img_set = f"Unlabeled_90.txt"
elif dataset_type == "CVC14":
    test.img_set = f"CVC-test.txt"

test.annotation = "AR-CNN"

test.input_size = [512., 640.]

### test model ~ eval.py
test.checkpoint = "./jobs/best_checkpoint.pth.tar"
test.batch_size = 4

### train_eval.py
test.eval_batch_size = 1


# KAIST Image Mean & STD
## RGB
IMAGE_MEAN = [0.3465,  0.3219,  0.2842]
IMAGE_STD = [0.2358, 0.2265, 0.2274]
## Lwir
LWIR_MEAN = [0.1598]
LWIR_STD = [0.0813]

                    
# dataset
dataset = edict()
dataset.workers = 4
dataset.OBJ_LOAD_CONDITIONS = {    
                                  'train': {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                                  'test': {'hRng': (-np.inf, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                              }


# Fusion Dead Zone
'''
Fusion Dead Zone
The input image of the KAIST dataset is input in order of [RGB, thermal].
Each case is as follows :
orignal, blackout_r, blackout_t, sidesblackout_a, sidesblackout_b, surroundingblackout
'''
FDZ_case = edict()

FDZ_case.original = ["None", "None"]

FDZ_case.blackout_r = ["blackout", "None"]
FDZ_case.blackout_t = ["None", "blackout"]

FDZ_case.sidesblackout_a = ["SidesBlackout_R", "SidesBlackout_L"]
FDZ_case.sidesblackout_b = ["SidesBlackout_L", "SidesBlackout_R"]
FDZ_case.surroundingblackout = ["None", "SurroundingBlackout"]


# main
args = edict(path=PATH,
             soft_teacher=soft_teacher,
             train=train,
             test=test,
             dataset=dataset,
             FDZ_case=FDZ_case)

args.dataset_type = dataset_type

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.exp_time = None
args.exp_name = None

args.n_classes = 3

## Semi Unpaired Augmentation
args.upaired_augmentation = ["TT_RandomHorizontalFlip",
                             "TT_FixedHorizontalFlip",
                             "TT_RandomResizedCrop"]

args.want_augmentation = ["RandomHorizontalFlip",
                          "FixedHorizontalFlip",
                          "RandomResizedCrop"]

args.same_augmentation = ["RandomHorizontalFlipForST"]

## Train dataset transform
args["train"].img_transform = Compose([ ColorJitter(0.3, 0.3, 0.3), 
                                        ColorJitterLWIR(contrast=0.3)
                                        
                                                                ])
args["train"].co_transform = Compose([   
                                        RandomHorizontalFlip(p=0.5), 
                                        RandomResizedCrop([512,640], \
                                                                scale=(0.25, 4.0), \
                                                                ratio=(0.8, 1.2)),
                                        ToTensor(), \
                                        Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                        Normalize(LWIR_MEAN, LWIR_STD, 'T') ], \
                                        args=args)
"""
TT_RandomHorizontalFlip(p=0.5), 
                                        TT_RandomResizedCrop([512,640], \
                                                                scale=(0.25, 4.0), \
                                                                ratio=(0.8, 1.2)),
"""
## Test dataset transform
args["test"].img_transform = Compose([ ])    
args["test"].co_transform = Compose([Resize(test.input_size), \
                                     ToTensor(), \
                                     Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                     Normalize(LWIR_MEAN, LWIR_STD, 'T')                        
                                    ])

## for soft teacher
args["train"].weak_transform = ComposeForST([ RandomHorizontalFlipForST(p=0.5),
                                         ToTensor(),
                                         Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), 
                                         Normalize(LWIR_MEAN, LWIR_STD, 'T')   
                                        ], args=args)
args["train"].strong_transform = ComposeForST([ ColorJitter(0.3, 0.3, 0.3), 
                                                ColorJitterLWIR(contrast=0.3),
                                                RandomHorizontalFlipForST(p=0.5),
                                                RandomErasing(),
                                                ToTensor(),
                                                Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), 
                                                Normalize(LWIR_MEAN, LWIR_STD, 'T'), 
                                            ], args=args)

# args["train"].general_flip = ComposeForST([ RandomHorizontalFlip(p=0.5) ])
args["train"].batch_weak_transform = ComposeForST([RandomHorizontalFlipForST(p=0.5),
                                         ToTensor(),
                                         Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), 
                                         Normalize(LWIR_MEAN, LWIR_STD, 'T')   
                                        ], args=args)
args["train"].batch_strong_transform = ComposeForST([RandomHorizontalFlipForST(p=0.5),
                                                ColorJitter(0.3, 0.3, 0.3), 
                                                ColorJitterLWIR(contrast=0.3),
                                                ToTensor(),
                                                RandomErasing(),
                                                Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), 
                                                Normalize(LWIR_MEAN, LWIR_STD, 'T'), 
                                            ], args=args)
args["train"].baseline_transform = ComposeForST([ RandomHorizontalFlipForST(p=0.5),
                                                ColorJitter(0.3, 0.3, 0.3), 
                                                ColorJitterLWIR(contrast=0.3),
                                                RandomResizedCrop([512,640], \
                                                                scale=(0.25, 4.0), \
                                                                ratio=(0.8, 1.2)),
                                                ToTensor(),
                                                Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), 
                                                Normalize(LWIR_MEAN, LWIR_STD, 'T'), 
                                            ], args=args)

ema = edict()
ema.use_scheduler = False
ema.tau = 0.0004 # it also be start_tau when using scheduler
ema.scheduling_start_epoch = 0
ema.max_tau = 0.01
ema.min_tau = 0.0001
