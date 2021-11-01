#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolo4 import YoloBody
from nets.yolo_training import LossHistory, YOLOLoss, weights_init
from utils.dataloader import YoloDataset, yolo_dataset_collate

from dataLoalderForEval import YoloDatasetForEval
from utils.metrics import ap_per_class
from utils.utils import DecodeBox
from utils.utils import non_max_suppression

import time
#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def compute_statistics(net,gen,nms_confidence_thres=0.5,nms_iou_thres=0.4):
    net.eval()

    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    stats = []
    p=[]
    r=[]
    ap=[]
    ap_class = []
    map50=0
    map=0

    inference_time = 0.0

    for iteration, batch in enumerate(tqdm(gen)):
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
            targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]

            torch.cuda.synchronize()
            time_start = time.time()

            outputs = net(images)
            output_list = []
            for i in range(4):
                output_list.append(yolo_decodes[i](outputs[i]))

            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(class_names),
                                                   conf_thres=nms_confidence_thres,
                                                   nms_thres=nms_iou_thres)
            torch.cuda.synchronize()
            time_end = time.time()
            inference_time +=time_end-time_start

        for si, pred in enumerate(batch_detections):
            labels = targets[0]
            nl = labels.size()[0]

            tcls = labels[:, -1].tolist() if nl else []  # target class

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)

            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, -1]

                tbox = labels[:, 0:4]
                tbox[:, 0] = tbox[:, 0] * input_shape[1]
                tbox[:, 1] = tbox[:, 1] * input_shape[0]
                tbox[:, 2] = tbox[:, 2] * input_shape[1]
                tbox[:, 3] = tbox[:, 3] * input_shape[0]

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, -1]).nonzero(as_tuple=False).view(-1)

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                # print("ious[j] = ",ious[j],"  iouv = ",iouv,ious[j] > iouv )
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), (pred[:, 4] * pred[:, 5]).cpu(), pred[:, -1].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        # print(p, r, ap, f1, ap_class)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    fps = 1/(inference_time/len(gen))
    return p,r,ap,map50,map,ap_class,fps                 #pr, recall,ap,map0.50,map0.95,class,fps


def fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    if Tensorboard:
        global train_tensorboard_step, val_tensorboard_step
    total_loss = 0
    val_loss = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size,desc=r'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for i in range(4):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            total_loss += loss.item()

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            if Tensorboard:
                # 将loss写入tensorboard，每一步都写
                writer.add_scalar('Train_loss', loss, train_tensorboard_step)
                train_tensorboard_step += 1

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    # 将loss写入tensorboard，下面注释的是每个世代保存一次
    # if Tensorboard:
    #     writer.add_scalar('Train_loss', total_loss/(iteration+1), epoch)
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=r'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(4):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

            # 将loss写入tensorboard, 下面注释的是每一步都写
            # if Tensorboard:
            #     writer.add_scalar('Val_loss', loss, val_tensorboard_step)
            #     val_tensorboard_step += 1
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)



    # 将loss写入tensorboard，每个世代保存一次
    if Tensorboard:
        writer.add_scalar('Val_loss',val_loss / (epoch_size_val+1), epoch)
    loss_history.append_loss(total_loss/(epoch_size+1), val_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Tensorboard
    #-------------------------------#
    Tensorboard = True
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda = True
    #------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    #------------------------------------------------------#
    normalize = True
    #-------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    #-------------------------------#
    input_shape = (608,608)
    #----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #----------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/new_classes.txt'
    #------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False 
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    #------------------------------------------------------#
    mosaic = True
    Cosine_lr = True
    smoooth_label = 0

    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    #------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改classes_path和对应的txt文件
    #------------------------------------------------------#
    model = YoloBody(len(anchors[0]), num_classes)
    weights_init(model)

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    model_path = "model_data/yolo4_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    yolo_loss    = YOLOLoss(np.reshape(anchors,[-1,2]), num_classes, (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize)
    yolo_loss    = YOLOLoss(np.reshape(anchors,[-1,2]), num_classes, (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize)
    loss_history = LossHistory("logs/")

    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    if Tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir='logs',flush_secs=60)
        if Cuda:
            graph_inputs = torch.randn(1,3,input_shape[0],input_shape[1]).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.randn(1,3,input_shape[0],input_shape[1]).type(torch.FloatTensor)
        writer.add_graph(model, graph_inputs)
        train_tensorboard_step  = 1
        val_tensorboard_step    = 1

    # ---------------------------------------------------#
    #   建立四个特征层解码用的工具
    # ---------------------------------------------------#
    yolo_decodes = []
    for i in range(4):
        yolo_decodes.append(
            DecodeBox(anchors[i], len(class_names), (input_shape[1], input_shape[0])))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 1e-3
        Batch_size      = 4
        Init_Epoch      = 0
        Freeze_Epoch    = 25
        
        #----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        #----------------------------------------------------------------------------#
        optimizer       = optim.Adam(net.parameters(),lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)

        train_dataset_for_statistics = YoloDatasetForEval(lines[:num_train], (input_shape[0], input_shape[1]))
        val_dataset_for_statistics =  YoloDatasetForEval(lines[num_train:], (input_shape[0], input_shape[1]))

        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)

        gen_for_statistic = DataLoader(train_dataset_for_statistics, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val_for_statistic = DataLoader(val_dataset_for_statistics, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size
        
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()
            p, r, ap, map50, map, ap_class, fps = compute_statistics(net, gen_for_statistic, nms_confidence_thres=0.5,nms_iou_thres=0.4)
            print("Train_map: ",map50)
            if Tensorboard:
                for i, pp in enumerate(p):
                    writer.add_scalar('Train_precision_' + class_names[ap_class[i]], p[i], epoch)
                    writer.add_scalar('Train_recall_' + class_names[ap_class[i]], r[i], epoch)
                    writer.add_scalar('Train_AP_' + class_names[ap_class[i]], ap[i], epoch)
                writer.add_scalar('Train_map0.50', map50, epoch)
                writer.add_scalar('Train_map0.95', map, epoch)

            p, r, ap, map50, map, ap_class, fps = compute_statistics(net, gen_val_for_statistic, nms_confidence_thres=0.5,nms_iou_thres=0.4)
            print("Val_map: ",map50)
            if Tensorboard:
                for i, pp in enumerate(p):
                    writer.add_scalar('Val_precision_' + class_names[ap_class[i]], p[i], epoch)
                    writer.add_scalar('Val_recall_' + class_names[ap_class[i]], r[i], epoch)
                    writer.add_scalar('Val_AP_' + class_names[ap_class[i]], ap[i], epoch)
                writer.add_scalar('Val_map0.50', map50, epoch)
                writer.add_scalar('Val_map0.95', map, epoch)
                writer.add_scalar('Inference_fps', fps, epoch)

    if True:
        lr              = 1e-4
        Batch_size      = 2
        Freeze_Epoch    = 25
        Unfreeze_Epoch  = 50

        #----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        #----------------------------------------------------------------------------#
        optimizer       = optim.Adam(net.parameters(),lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)

        train_dataset_for_statistics = YoloDatasetForEval(lines[:num_train], (input_shape[0], input_shape[1]))
        val_dataset_for_statistics =  YoloDatasetForEval(lines[num_train:], (input_shape[0], input_shape[1]))

        gen_for_statistic = DataLoader(train_dataset_for_statistics, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val_for_statistic = DataLoader(val_dataset_for_statistics, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size
        
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step()
            # ------------------------------------------#
            # 计算mAP recall pr AP
            # ------------------------------------------#
            p, r, ap, map50, map, ap_class, fps = compute_statistics(net, gen_for_statistic, nms_confidence_thres=0.5,nms_iou_thres=0.4)
            print("Train_map: ",map50)
            if Tensorboard:
                for i, pp in enumerate(p):
                    writer.add_scalar('Train_precision_' + class_names[ap_class[i]], p[i], epoch)
                    writer.add_scalar('Train_recall_' + class_names[ap_class[i]], r[i], epoch)
                    writer.add_scalar('Train_AP_' + class_names[ap_class[i]], ap[i], epoch)
                writer.add_scalar('Train_map0.50', map50, epoch)
                writer.add_scalar('Train_map0.95', map, epoch)

            p, r, ap, map50, map, ap_class, fps = compute_statistics(net, gen_val_for_statistic ,nms_confidence_thres=0.5,nms_iou_thres=0.4)
            print("Val_map: ",map50)
            if Tensorboard:
                for i, pp in enumerate(p):
                    writer.add_scalar('Val_precision_' + class_names[ap_class[i]], p[i], epoch)
                    writer.add_scalar('Val_recall_' + class_names[ap_class[i]], r[i], epoch)
                    writer.add_scalar('Val_AP_' + class_names[ap_class[i]], ap[i], epoch)
                writer.add_scalar('Val_map0.50', map50, epoch)
                writer.add_scalar('Val_map0.95', map, epoch)
                writer.add_scalar('Inference_fps', fps, epoch)
