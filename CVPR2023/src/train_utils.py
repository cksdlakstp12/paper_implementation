from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pickle
from collections import defaultdict

from model import SSD300, SSD300_3Way

def initialize_state(n_classes, train_conf):
    model = SSD300(n_classes=n_classes)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    return model, optimizer, optim_scheduler, train_conf.start_epoch, None

def load_state_from_checkpoint(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_from_checkpoint_with_new_optim(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    
    return model, optimizer, optim_scheduler, start_epoch, None
    
def load_state(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state(args.n_classes, train_conf)
    else:
        if "teacher_weights.pth.tar071" in checkpoint:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_with_new_optim(train_conf, checkpoint)
        else:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint(train_conf, checkpoint)

    return (model, optimizer, optim_scheduler, start_epoch, train_loss)

def load_SoftTeacher(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state(config, student_checkpoint)
    teacher_state = load_state(config, teacher_checkpoint)

    return *student_state, *teacher_state

def initialize_state_3way(n_classes, train_conf):
    model = SSD300_3Way(n_classes=n_classes)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    return model, optimizer, optim_scheduler, train_conf.start_epoch, None

def load_state_from_checkpoint_3way(n_classes, train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_from_checkpoint_3way_with_new_optim(n_classes, train_conf, checkpoint):
    init_model, optimizer, optim_scheduler, train_conf.start_epoch, _ = initialize_state_3way(n_classes, train_conf)

    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    pret_model = checkpoint['model']
    
    for pret_name, pret_param in pret_model.named_parameters():
        if "pred_convs" in pret_name:
            for name_pred in ["pred_convs", "pred_convs_vis", "pred_convs_lwir"]:
                seach_name = pret_name.replace("pred_convs", name_pred)
                init_param = dict(init_model.named_parameters())[seach_name]
                init_param.data.copy_(pret_param.data)
        elif pret_name in dict(pret_model.named_parameters()):
            init_param = dict(init_model.named_parameters())[pret_name]
            init_param.data.copy_(pret_param.data)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in init_model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)

    return init_model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_3way(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state_3way(args.n_classes, train_conf)
    else:
        if "teacher_weights.pth.tar071" in checkpoint:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_3way_with_new_optim(args.n_classes, train_conf, checkpoint)
        else:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_3way(args.n_classes, train_conf, checkpoint)

    return (model, optimizer, optim_scheduler, start_epoch, train_loss)

def load_SoftTeacher_3way(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state_3way(config, student_checkpoint)
    teacher_state = load_state_3way(config, teacher_checkpoint)

    return *student_state, *teacher_state

def create_dataloader(config, dataset_class, sample_mode = None, **kwargs):
    if kwargs["condition"] == "train":
        if sample_mode == "two":
            sample = "Labeled"
            dataset = dataset_class(config.args, sample = sample,**kwargs)
            L_loader = DataLoader(dataset, batch_size=int(config.train.batch_size/2), shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
            sample = "Unlabeled"
            dataset = dataset_class(config.args, sample = sample, **kwargs)
            U_loader = DataLoader(dataset, batch_size=int(config.train.batch_size/2), shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
            return dataset, L_loader, U_loader
        else: 
            dataset = dataset_class(config.args, **kwargs)
            loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
    else:
        dataset = dataset_class(config.args, **kwargs)
        test_batch_size = config.args["test"].eval_batch_size * torch.cuda.device_count()
        loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False,
                              num_workers=config.dataset.workers,
                              collate_fn=dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here
    return dataset, loader

def converter(originpath, changepath, wantname):
    # Loading the 90percents.txt file and creating a dictionary where keys are the index
    with open("./imageSets/" + originpath, 'r') as f:
        data_90 = {idx+1: line.strip() for idx, line in enumerate(f)}

    # Loading the test2.txt file
    with open(changepath, 'r') as f:
        data_test2 = f.readlines()

    # Replacing the first number of each line in test2.txt with corresponding line in 90percents.txt
    data_test2_new = []
    for line in data_test2:
        items = line.split(',')
        index = int(items[0])
        items[0] = data_90[index]
        data_test2_new.append(','.join(items))

    # Writing the new data into a new file
    with open(wantname, 'w') as f:
        for line in data_test2_new:
            f.write(line)

def soft_update(teacher_model, student_model, tau):
    """
    Soft update model parameters.
    θ_teacher = τ*θ_student + (1 - τ)*θ_teacher

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    :param tau: interpolation parameter (0.001 in your case)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(tau*student_param.data + (1.0-tau)*teacher_param.data)

def copy_student_to_teacher(teacher_model, student_model):
    """
    Copy student model to teacher model.
    θ_teacher = θ_student

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(student_param.data)

class EMAScheduler():
    def __init__(self, config):
        self.use_scheduler = config.ema.use_scheduler
        self.start_tau = config.ema.tau
        self.scheduling_start_epoch = config.ema.scheduling_start_epoch
        self.max_tau = config.ema.max_tau
        self.min_tau = config.ema.min_tau
        self.last_tau = config.ema.tau

    @staticmethod
    def calc_tau(epoch, tau):
        tau = tau
        return tau
    
    def get_tau(self, epoch):
        if not self.use_scheduler:
            return self.start_tau
        else:
            new_tau = EMAScheduler.calc_tau(epoch, self.last_tau)
            return new_tau
            """
            if new_tau > self.max_tau: 
                return self.max_tau
            elif new_tau < self.min_tau: 
                return self.min_tau
            else:
                self.last_tau = new_tau 
                return new_tau
            """

def load_KMeans_model(model_path):
    loaded_kmeans = load(model_path)
    return loaded_kmeans

def load_mean_std_dict(dict_path):
    with open(dict_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def translate_coordinate(box, feature_w, feature_h, ori_w, ori_h, is_GT):
    x, y, w, h = box
    new_x = int(x * feature_w) if is_GT else int(x / ori_w * feature_w)
    new_y = int(y * feature_h) if is_GT else int(y / ori_h * feature_h)
    new_w = int(w * feature_w) if is_GT else int(w / ori_w * feature_w)
    new_h = int(h * feature_h) if is_GT else int(h / ori_h * feature_h)
    return new_x, new_y, new_w, new_h

def joint_confidence_estimation(class_scores, iou_scores):
    return class_scores * iou_scores

def teacher_student_agreement(teacher_scores, iou_scores, threshold=0.5):
    # JCE를 기반으로 긍정적 및 부정적 예측 인덱스 결정
    joint_confidence = joint_confidence_estimation(teacher_scores, iou_scores)
    positive_indices = torch.where(joint_confidence < threshold)[0]
    negative_indices = torch.where(joint_confidence >= threshold)[0]
    return positive_indices, negative_indices

def iou(boxA, boxB):
    # boxA와 boxB의 교차 영역 계산
    xA = torch.max(boxA[0], boxB[0])
    yA = torch.max(boxA[1], boxB[1])
    xB = torch.min(boxA[2], boxB[2])
    yB = torch.min(boxA[3], boxB[3])

    # 교차 영역의 면적 계산
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 각 박스의 면적 계산
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU 계산
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def batch_iou(boxes1, boxes2):
    """
    벡터화된 IoU 계산 함수.
    boxes1: [N, 4] 크기의 텐서 (N개의 박스, 각 박스는 [x1, y1, x2, y2] 형식)
    boxes2: [M, 4] 크기의 텐서 (M개의 박스, 각 박스는 [x1, y1, x2, y2] 형식)
    결과: [N, M] 크기의 IoU 행렬
    """

    # 각 박스 좌표 확장 및 비교를 위한 재배열
    # boxes1은 [N, 1, 4], boxes2는 [1, M, 4]로 변경
    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]

    # 교차 영역의 좌표 계산
    max_xy = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    min_xy = torch.max(boxes1[..., :2], boxes2[..., :2])

    # 교차 영역의 면적 계산
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]

    # 각 박스의 면적 계산
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # IoU 계산
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area

    return iou

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    return F_loss.mean()

def bce_loss(inputs, targets):
    return F.binary_cross_entropy_with_logits(inputs, targets)

def calculate_box_loss(iou_scores, predicted_scores, alpha=0.25, gamma=2.0):
    soft_labels = iou_scores  # IoU 기반 소프트 레이블

    # Focal Loss 계산
    focal_loss_val = focal_loss(predicted_scores, soft_labels, alpha, gamma)

    # BCE Loss 계산
    bce_loss_val = bce_loss(predicted_scores, soft_labels)

    # 총 박스 손실
    total_loss = focal_loss_val + bce_loss_val
    return total_loss

def calculate_unsupervised_loss(teacher_scores, student_locs, student_scores, pseudo_boxes, pseudo_labels):
    batch_size, num_boxes, _ = student_locs.size()
    iou_scores = torch.zeros(batch_size, num_boxes, 3, device=student_locs.device)

    print("student_scores.shape : ", student_scores.shape)
    print("pseudo_boxes : ", pseudo_boxes)
    print("pseudo_labels : ", pseudo_labels)

    for batch_idx in range(batch_size):
        # pseudo_boxes[batch_idx]가 단일 텐서인 경우 리스트로 변환
        if isinstance(pseudo_boxes[batch_idx], torch.Tensor):
            pseudo_boxes_tensor = pseudo_boxes[batch_idx].unsqueeze(0)  # [1, 4]
        else:
            pseudo_boxes_tensor = torch.stack(pseudo_boxes[batch_idx]).to(student_locs.device)  # [num_pseudo_boxes, 4]
        
        iou_matrix = batch_iou(student_locs[batch_idx], pseudo_boxes_tensor)  # [41760, num_pseudo_boxes]
        max_iou, _ = torch.max(iou_matrix, dim=1)  # [41760]

        # max_iou를 [41760, 3]으로 확장
        if max_iou.dim() == 1:
            max_iou = max_iou.unsqueeze(1)  # [41760, 1]
        max_iou_expanded = max_iou.expand(-1, 3)  # [41760, 3]
        iou_scores[batch_idx, :, :] = max_iou_expanded

    # TSA 계산
    pos_indices_tuple, neg_indices_tuple = teacher_student_agreement(teacher_scores, iou_scores)

    unsupervised_loss = 0
    student_scores_flat = student_scores.view(-1, 3)

    for batch_idx in range(batch_size):
        current_labels = pseudo_labels[batch_idx]
        current_scores = student_scores_flat[batch_idx * num_boxes:(batch_idx + 1) * num_boxes]

        # positive_indices 처리
        if pos_indices_tuple[0].dim() > 0:
            pos_indices_batch = pos_indices_tuple[0][pos_indices_tuple[0][:, 0] == batch_idx][:, 1]
            valid_pos_indices = current_labels[pos_indices_batch] != -1
            pos_scores = current_scores[pos_indices_batch][valid_pos_indices]
            pos_labels = current_labels[pos_indices_batch][valid_pos_indices].long()
            unsupervised_loss += F.cross_entropy(pos_scores, pos_labels)

        # negative_indices 처리
        if neg_indices_tuple[0].dim() > 0:
            neg_indices_batch = neg_indices_tuple[0][neg_indices_tuple[0][:, 0] == batch_idx][:, 1]
            valid_neg_indices = current_labels[neg_indices_batch] != -1
            neg_scores = current_scores[neg_indices_batch][valid_neg_indices]
            neg_labels = current_labels[neg_indices_batch][valid_neg_indices].long()
            unsupervised_loss += F.cross_entropy(neg_scores, neg_labels)

    return unsupervised_loss