#!/usr/bin/env python
# coding: utf-8

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[1]:


get_ipython().system('pwd')
get_ipython().system('pip install scikit-image')
get_ipython().system('pip install pydicom')
get_ipython().system('pip install paddleseg')


# In[ ]:


get_ipython().system(' git clone https://gitee.com/paddlepaddle/PaddleSeg.git')
get_ipython().run_line_magic('cd', '~/PaddleSeg/')
get_ipython().system(' pip install paddleseg')


# In[ ]:


get_ipython().system(' mkdir dataset')
get_ipython().run_line_magic('cd', 'dataset')
get_ipython().system('unzip -q -n /home/aistudio/data/data144729/RVSC.zip  -d /home/aistudio/PaddleSeg/dataset')


# In[2]:


import pandas as pd
import os
from tqdm import tqdm
import logging
import numpy as np

from PIL import Image

import cv2
import pydicom
import pydicom
import matplotlib.pyplot as plt
import scipy.misc


# In[3]:


import os
import numpy as np
import cv2
import pydicom
import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import random

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")

def create_datalist(data_path,image_path,label_path):
    data_names = os.listdir(image_path)
    random.shuffle(data_names)  # 打乱数据
    k=0
    with open(os.path.join(data_path, 'train_list.txt'), 'w') as tf:
        with open(os.path.join(data_path, 'val_list.txt'), 'w') as vf:
            for i in data_names:
                patient_img=image_path+'/'+i
                label_img=label_path+'/'+i
                for j in os.listdir(patient_img):
                    # print(patient_img+'/'+j)
                    # print(label_img+'/'+j)
                    if k%9==0:
                        vf.write(patient_img+'/'+j + ' ' + label_img+'/'+j + '\n')
                    else:
                        tf.write(patient_img+'/'+j + ' ' + label_img+'/'+j + '\n')
                    k=k+1


def MakeDataset(train_path,label_path,data_path,label_path_in,label_path_out,model='train'):
    mkdir(label_path_out)
    mkdir(label_path_in)
    mkdir(data_path)
    for i in os.listdir(train_path):
        # print(i)
        patient=i[7:11]
        file=train_path+'/patient'+str(patient)+'/P'+str(patient)+'list.txt'
        # print(file)
        names=[]
        with open(file) as f:
            line = f.readline()
            while line:
                # print(line.split('\\')[3][0:8])
                names.append(line.split('\\')[3][0:8])
                # line_float=[float(line.split()[0]),float(line.split()[1])]
                # array_in.append(line_float)
                line = f.readline()
                line = f.readline()
        for name in names:
            patient=name[1:3]
            idx=name[4:10]
            # print(patient,idx)
            
            path=train_path+'/patient'+str(patient)+'/P'+str(patient)+'dicom/P'+str(patient)+'-'+str(idx)+'.dcm'
            if(model=='train'):
                in_path=label_path+'/patient'+str(patient)+'/P'+str(patient)+'contours-manual/P'+str(patient)+'-'+str(idx)+'-icontour-manual.txt'
                out_path=label_path+'/patient'+str(patient)+'/P'+str(patient)+'contours-manual/P'+str(patient)+'-'+str(idx)+'-ocontour-manual.txt'
            else:
                in_path=label_path+'/P'+str(patient)+'contours-manual/P'+str(patient)+'-'+str(idx)+'-icontour-manual.txt'
                out_path=label_path+'/P'+str(patient)+'contours-manual/P'+str(patient)+'-'+str(idx)+'-ocontour-manual.txt'
            ds = pydicom.read_file(path)  #读取.dcm文件
            img = ds.pixel_array  # 提取图像信息
            
            # plt.axis('off')
            # plt.imshow(img)
            
            # plt.savefig(data_path+'/P'+str(patient)+'-'+str(idx)+'.png',bbox_inches='tight')
            cv2.imwrite(data_path+'/P'+str(patient)+'-'+str(idx)+'.png',img*255)
            # plt.close()
            array_in=[]
            array_out=[]
            with open(in_path) as f:
                line = f.readline()
                while line:
                        line_float=[float(line.split()[0]),float(line.split()[1])]
                        array_in.append(line_float)
                        line = f.readline()
                        
            with open(in_path) as f:
                line = f.readline()
                while line:
                    line_float=[float(line.split()[0]),float(line.split()[1])]
                    array_out.append(line_float)
                    line = f.readline()
            array_in=np.array(array_in)
            array_out=np.array(array_out)
            # print(type(array_in))
            imgName=data_path+'/P'+str(patient)+'-'+str(idx)+'.png'
            # 展示原图
            img_mask = cv2.imread(imgName)
            # plt.imshow(img_mask)
            plt.axis('off')
            # 创建掩膜
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            polygon_in = np.array(array_in, np.int32) # 坐标为顺时针方向
            cv2.fillConvexPoly(mask, polygon_in, (255, 255, 255))
            # # 展示掩膜图片
            # plt.imshow(mask)
            cv2.imwrite(label_path_in+'/P'+str(patient)+'-'+str(idx)+'.png',mask)
            img=cv2.imread(label_path_in+'/P'+str(patient)+'-'+str(idx)+'.png')
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            difference = (img_gray.max() - img_gray.min()) // 2
            _, img_binary = cv2.threshold(img_gray, difference, 1, cv2.THRESH_BINARY)
            cv2.imwrite(label_path_in+'/P'+str(patient)+'-'+str(idx)+'.png',img_binary)            
            # plt.savefig(label_path_in+'/P'+str(patient)+'-'+str(idx)+'.png',bbox_inches='tight')
            # plt.close()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            polygon_out = np.array(array_out, np.int32) # 坐标为顺时针方向
            cv2.fillConvexPoly(mask, polygon_in, (255, 255, 255))

            # plt.axis('off')
            # plt.imshow(mask)
            # plt.savefig(label_path_out+'/P'+str(patient)+'-'+str(idx)+'.png',bbox_inches='tight')
            cv2.imwrite(label_path_out+'/P'+str(patient)+'-'+str(idx)+'.png',mask)
            img=cv2.imread(label_path_out+'/P'+str(patient)+'-'+str(idx)+'.png')
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            difference = (img_gray.max() - img_gray.min()) // 2
            _, img_binary = cv2.threshold(img_gray, difference, 1, cv2.THRESH_BINARY)
            cv2.imwrite(label_path_out+'/P'+str(patient)+'-'+str(idx)+'.png',img_binary)
            # plt.close()


# In[ ]:


train_path='/home/aistudio/PaddleSeg/dataset/RVSC/TrainingSet'
label_path='/home/aistudio/PaddleSeg/dataset/RVSC/TrainingSet'
data_path='/home/aistudio/PaddleSeg/dataset/train_img'
label_path_in='/home/aistudio/PaddleSeg/dataset/train_label_img_in'
label_path_out='/home/aistudio/PaddleSeg/dataset/train_label_img_out'
MakeDataset(train_path,label_path,data_path,label_path_in,label_path_out,'train')
train_path='/home/aistudio/PaddleSeg/dataset/RVSC/Test1Set'
label_path='/home/aistudio/PaddleSeg/dataset/RVSC/Test1SetContours/'
data_path='/home/aistudio/PaddleSeg/dataset/test1_img'
label_path_in='/home/aistudio/PaddleSeg/dataset/test1_label_img_in'
label_path_out='/home/aistudio/PaddleSeg/dataset/test1_label_img_out'
MakeDataset(train_path,label_path,data_path,label_path_in,label_path_out,'val')

train_path='/home/aistudio/PaddleSeg/dataset/RVSC/Test2Set'
label_path='/home/aistudio/PaddleSeg/dataset/RVSC/Test2SetContours/'
data_path='/home/aistudio/PaddleSeg/dataset/test2_img'
label_path_in='/home/aistudio/PaddleSeg/dataset/test2_label_img_in'
label_path_out='/home/aistudio/PaddleSeg/dataset/test2_label_img_out'
MakeDataset(train_path,label_path,data_path,label_path_in,label_path_out,'val')


# In[4]:


get_ipython().run_line_magic('cd', '/home/aistudio/PaddleSeg/dataset/')
import random
get_ipython().system('pwd')
data_path=''
image_path='train_img'
label_path='train_label_img_in'
# create_datalist(data_path,image_path,label_path)
data_names = os.listdir(image_path)
random.shuffle(data_names)  # 打乱数据
k=0
with open(os.path.join(data_path, 'train_list.txt'), 'w') as tf:
    with open(os.path.join(data_path, 'val_list.txt'), 'w') as vf:
        for i in data_names:
            # print(i)
            patient_img=image_path+'/'+i
            label_img=label_path+'/'+i
            
            if k%9==0:
                vf.write(patient_img+ ' ' + label_img+ '\n')
            else:
                tf.write(patient_img + ' ' + label_img + '\n')
            k=k+1
data_path=''
image_path='test1_img'
label_path='test1_label_img_in'
data_names = os.listdir(image_path)
random.shuffle(data_names)  # 打乱数据
with open(os.path.join(data_path, 'test1_list.txt'), 'w') as tf:
    for i in data_names:
        patient_img=image_path+'/'+i
        label_img=label_path+'/'+i
        tf.write(patient_img +'\n')

data_path=''
image_path='test2_img'
label_path='test2_label_img_in'
data_names = os.listdir(image_path)
random.shuffle(data_names)  # 打乱数据
with open(os.path.join(data_path, 'test2_list.txt'), 'w') as tf:
    for i in data_names:
        patient_img=image_path+'/'+i
        label_img=label_path+'/'+i
        tf.write(patient_img+'\n')


# In[5]:


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class BCELoss(nn.Layer):

    def __init__(self,
                 weight=None,
                 pos_weight=None,
                 ignore_index=255,
                 edge_label=False):
        super().__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index
        self.edge_label = edge_label
        self.EPS = 1e-10

        if self.weight is not None:
            if isinstance(self.weight, str):
                if self.weight != 'dynamic':
                    raise ValueError(
                        "if type of `weight` is str, it should equal to 'dynamic', but it is {}"
                        .format(self.weight))
            elif not isinstance(self.weight, paddle.Tensor):
                raise TypeError(
                    'The type of `weight` is wrong, it should be Tensor or str, but it is {}'
                    .format(type(self.weight)))

    def forward(self, logit, label):
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)
        mask = (label != self.ignore_index)
        mask = paddle.cast(mask, 'float32')
        # label.shape should equal to the logit.shape
        if label.shape[1] != logit.shape[1]:
            label = label.squeeze(1)
            label = F.one_hot(label, logit.shape[1])
            label = label.transpose((0, 3, 1, 2))
        if isinstance(self.weight, str):
            pos_index = (label == 1)
            neg_index = (label == 0)
            pos_num = paddle.sum(pos_index.astype('float32'))
            neg_num = paddle.sum(neg_index.astype('float32'))
            sum_num = pos_num + neg_num
            weight_pos = 2 * neg_num / (sum_num + self.EPS)
            weight_neg = 2 * pos_num / (sum_num + self.EPS)
            weight = weight_pos * label + weight_neg * (1 - label)
        else:
            weight = self.weight
        if isinstance(self.pos_weight, str):
            pos_index = (label == 1)
            neg_index = (label == 0)
            pos_num = paddle.sum(pos_index.astype('float32'))
            neg_num = paddle.sum(neg_index.astype('float32'))
            sum_num = pos_num + neg_num
            pos_weight = 2 * neg_num / (sum_num + self.EPS)
        else:
            pos_weight = self.pos_weight
        label = label.astype('float32')
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit,
            label,
            weight=weight,
            reduction='none',
            pos_weight=pos_weight)
        loss = loss * mask
        loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)
        label.stop_gradient = True
        mask.stop_gradient = True

        return loss


# In[6]:


get_ipython().run_line_magic('cd', '/home/aistudio/PaddleSeg/')
# 参数调整
import paddle
from paddleseg.models import UNet, UNetPlusPlus
# from unet import UNet_3Plus
import paddleseg.transforms as T
from paddleseg.datasets import Dataset
# from paddleseg.models.losses import DiceLoss

# from paddleseg.models.losses import CrossEntropyLoss,FocalLoss,LovaszSoftmaxLoss
# 构建训练集
train_transforms = [
    T.RandomHorizontalFlip(),  # 水平翻转
    T.RandomVerticalFlip(),  # 垂直翻转
    T.RandomRotation(),  # 随机旋转
    T.RandomScaleAspect(),  # 随机缩放
    T.RandomDistort(),  # 随机扭曲
    T.Resize(target_size=(256, 256)),  # 这里为了加快速度，改为256x256
    T.Normalize()  # 归一化
]
train_dataset = Dataset(
    transforms=train_transforms,
    dataset_root='dataset',
    num_classes=2,
    mode='train',
    train_path='dataset/train_list.txt',
    separator=' ',
)
# 构建验证集
val_transforms = [
    T.Resize(target_size=(256, 256)),
    T.Normalize()
]
val_dataset = Dataset(
    transforms=val_transforms,
    dataset_root='dataset',
    num_classes=2,
    mode='val',
    val_path='dataset/val_list.txt',
    separator=' ',
)
# 优化器及损失
epochs = 5
batch_size = 16
# iters = epochs * 7278 // batch_size //2
iters=200
base_lr = 2e-3
losses = {}
# losses['types'] = [LovaszSoftmaxLoss()]
# losses['coef'] = [1]
losses['types'] = [BCELoss()] 
losses['coef'] = [1]


# In[9]:


# 重写train函数
import os
import time
from collections import deque
import shutil

import paddle
import paddle.nn.functional as F

from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, train_profiler, op_flops_funs)
from paddleseg.core.val import evaluate


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


# def loss_computation(logits, labels, losses, edges=None):
#    
#     loss_i = losses['types'][0]
#     coef_i = losses['coef'][0]

#     
#     if loss_i.__class__.__name__ == 'MixedLoss':
#         mixed_loss_list = loss_i(logits, labels)
#         
#         loss_list = [coef_i * mixed_loss for mixed_loss in mixed_loss_list]
#     elif loss_i.__class__.__name__ in ("KLLoss", ):
#         
#         loss_list = [coef_i * loss_i(logits_list[0], logits_list[1].detach())]
#     else:
#         
#         loss_list = [coef_i * loss_i(logits, labels)]

#     
#     return 

def loss_computation(logits_list, labels, losses, edges=None):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        coef_i = losses['coef'][i]

        if loss_i.__class__.__name__ == 'MixedLoss':
            mixed_loss_list = loss_i(logits, labels)
            for mixed_loss in mixed_loss_list:
                loss_list.append(coef_i * mixed_loss)
        elif loss_i.__class__.__name__ in ("KLLoss", ):
            loss_list.append(coef_i *
                             loss_i(logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(coef_i * loss_i(logits, labels))
    return loss_list


# 新增了一个参数 filename 就是保存的npy文件位置 位置就是save_dir+filename
def train(model,
          train_dataset,
          filename,
          val_dataset=None,
          optimizer=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          losses=None,
          keep_checkpoint_max=5,
          test_config=None,
          precision='fp32',
          amp_level='O1',
          profiler_options=None,
          to_static_training=False):
    filename=save_dir+'/'+filename
    print(filename)
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    start_iter = 0
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    # use amp
    if precision == 'fp16':
        logger.info('use AMP to train. AMP level = {}'.format(amp_level))
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        if amp_level == 'O2':
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level='O2',
                save_dtype='float32')

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        worker_init_fn=worker_init_fn, )

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    if to_static_training:
        model = paddle.jit.to_static(model)
        logger.info("Successfully to apply @to_static")

    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(batch_sampler)
    best_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()

    iter = start_iter
    loss_save_list=[]
    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                version = paddle.__version__
                if version == '2.1.2':
                    continue
                else:
                    break
            reader_cost_averager.record(time.time() - batch_start)
            images = data['img']
            labels = data['label'].astype('int64')
            # print(labels.shape)
            edges = None
            if len(data) == 3:
                edges = data[2].astype('int64')
            if hasattr(model, 'data_format') and model.data_format == 'NHWC':
                images = images.transpose((0, 2, 3, 1))

            
            logits_list = ddp_model(images) if nranks > 1 else model(images)
            loss_list = loss_computation(
                logits_list=logits_list,
                labels=labels,
                losses=losses,
                edges=edges)
            loss = sum(loss_list)
            # 这一步保存loss
            loss_save_list.append(loss)
            loss.backward()
            # if the optimizer is ReduceOnPlateau, the loss is the one which has been pass into step.
            if isinstance(optimizer, paddle.optimizer.lr.ReduceOnPlateau):
                optimizer.step(loss)
            else:
                optimizer.step()

            lr = optimizer.get_lr()

            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            train_profiler.add_profiler_step(profiler_options)

            model.clear_gradients()
            avg_loss += loss.numpy()[0]
            
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((iter - 1
                             ) // iters_per_epoch + 1, iter, iters, avg_loss,
                            lr, avg_train_batch_cost, avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter)
                    # Record all losses if there are more than 2 losses.
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)

                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)
                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (iter % save_interval == 0 or
                    iter == iters) and (val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0

                if test_config is None:
                    test_config = {}

                mean_iou, acc, _, _, _ = evaluate(
                    model,
                    val_dataset,
                    num_workers=num_workers,
                    precision=precision,
                    amp_level=amp_level,
                    **test_config)

                model.train()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                current_save_dir = os.path.join(save_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                save_models.append(current_save_dir)
                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if val_dataset is not None:
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        best_model_iter = iter
                        best_model_dir = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                        .format(best_mean_iou, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            batch_start = time.time()

    # Calculate flops.
    if local_rank == 0 and not (precision == 'fp16' and amp_level == 'O2'):
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()
    
    np.save(filename, loss_save_list)


# # **uent及其变种**

# In[10]:


unet_model = UNet(num_classes=2)


# In[11]:


get_ipython().system('export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡')
print(type(unet_model))
# paddle.summary(unet_model, (1, 3, 128, 128))  # 查看网络结构
lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)
unt_optimizer = paddle.optimizer.Adam(lr, parameters=unet_model.parameters(),weight_decay=4.0e-5)
# 训练
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
# 更改对应文件名即可
train(
    model=unet_model,
    filename='unet.npy',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=unt_optimizer,
    save_dir='output_unet',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/5),
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)


# # **AttentionUnet**

# In[12]:


from paddleseg.models import AttentionUNet

# 创建attention_unet模型
attentionunet_model = AttentionUNet(num_classes=2)

# 其余的训练设置保持不变...

# 为attention_unet创建优化器
attention_optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-3,
    parameters=attentionunet_model.parameters(),
    weight_decay=4.0e-5
)

# 训练attention_unet
train(
    model=attentionunet_model,
    filename='attentionunet.npy',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=attention_optimizer,
    save_dir='output_attentionunet',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/5),
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True
)


# # **Unet++**

# In[13]:


from paddleseg.models import UNet,UNetPlusPlus,UNet3Plus

unetpp_model = UNetPlusPlus(num_classes=2)

# unet++
optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-3,
    parameters=unetpp_model.parameters(),
    weight_decay=4.0e-5
)

train(
    model=unetpp_model,
    filename='unetpp.npy',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output_unet++',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/5),
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True
)


# # **Unet3+**

# In[14]:


unet3p_model = UNet3Plus(num_classes=2)

# unet3+
optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-3,
    parameters=unet3p_model.parameters(),
    weight_decay=4.0e-5
)

train(
    model=unet3p_model,
    filename='unet3p.npy',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output_unet3+',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/5),
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True
)


# In[21]:


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = ['U2Net', 'U2Netp']


@manager.MODELS.add_component
class U2Net(nn.Layer):
    """
    The U^2-Net implementation based on PaddlePaddle.

    The original article refers to
    Xuebin Qin, et, al. "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"
    (https://arxiv.org/abs/2005.09007).

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (int, optional): Input channels. Default: 3.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.

    """

    def __init__(self, num_classes=2, in_channels=3, pretrained=None):
        super(U2Net, self).__init__()

        self.stage1 = RSU7(in_channels, 32, 64)
        self.pool12 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2D(64, num_classes, 3, padding=1)
        self.side2 = nn.Conv2D(64, num_classes, 3, padding=1)
        self.side3 = nn.Conv2D(128, num_classes, 3, padding=1)
        self.side4 = nn.Conv2D(256, num_classes, 3, padding=1)
        self.side5 = nn.Conv2D(512, num_classes, 3, padding=1)
        self.side6 = nn.Conv2D(512, num_classes, 3, padding=1)

        self.outconv = nn.Conv2D(6 * num_classes, num_classes, 1)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(paddle.concat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(paddle.concat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(paddle.concat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(paddle.concat((hx2dup, hx1), 1))

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(paddle.concat((d1, d2, d3, d4, d5, d6), 1))
        # if d0.shape[1] > 1:
        #     d0 = d0[:, 0, :, :].unsqueeze(1)
        return [d0]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

class REBNCONV(nn.Layer):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2D(
            in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2D(out_ch)
        self.relu_s1 = nn.ReLU()

    def forward(self, x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):

    src = F.upsample(src, size=paddle.shape(tar)[2:], mode='bilinear')

    return src
### RSU-7 ###
class RSU7(nn.Layer):  #UNet07DRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(paddle.concat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(paddle.concat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(paddle.concat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(paddle.concat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Layer):  #UNet06DRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(paddle.concat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(paddle.concat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(paddle.concat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Layer):  #UNet05DRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(paddle.concat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(paddle.concat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Layer):  #UNet04DRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(paddle.concat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Layer):  #UNet04FRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(paddle.concat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(paddle.concat((hx2d, hx1), 1))

        return hx1d + hxin


# # **u2net**

# In[22]:



from paddle.nn import BCELoss  # 假设您使用的是二元交叉熵损失
# from paddleseg.models import U2Net
# losses = {
#     'types': [BCELoss(),BCELoss(),BCELoss(),BCELoss(),BCELoss(),BCELoss(),BCELoss()],
#     'coef': [1,1,1,1,1,1,1]  # 假设您对所有输出的损失都赋予相同的权重
# }

u2net_model = U2Net(num_classes=2)
batch_size = 1
# u2net
optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-3,
    parameters=u2net_model.parameters(),
    weight_decay=4.0e-5
)

train(
    model=u2net_model,
    filename='u2net.npy',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output_u2net',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/5),
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True
)


# # **注意力模型**

# In[23]:


import paddle
import paddle.nn as nn
class ChannelAttention(nn.Layer):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = nn.Conv2D(in_channels, in_channels // reduction_ratio, 1, bias_attr=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2D(in_channels // reduction_ratio, in_channels, 1, bias_attr=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out= paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv(x)
        return self.sigmoid(x)
class CBAM(nn.Layer):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


# In[24]:


class SEAttention(nn.Layer):
    def __init__(self, in_channels, hidden_channels=None):
        super(SEAttention, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = in_channels // 8
        
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Conv2D(in_channels, hidden_channels, kernel_size=1, bias_attr=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(hidden_channels, in_channels, kernel_size=1, bias_attr=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


# In[25]:


# 加入注意力机制之后，重写unet，形成senet-unet和cbam-unet
# 可以参考SE与CBAM注意力机制改进的U-Net_副本，之前上课的一个练习

# 编码器
# 其中nn.Layer表示该类继承自PaddlePaddle的深度学习网络层（layer），
# 表明Encoder本质上是一个神经网络模型。
class Encoder(nn.Layer):
    def __init__(self, num_channels, num_filters,attention=False):
        # super(Encoder, self)调用Encoder类的父类，并且将当前实例对象
        # self作为参数传递给它，以便在父类中进行相应的初始化工作。然后通过__init__()
        # 方法完成Encoder这个子类自己的初始化工作。
        super(Encoder,self).__init__()
        if attention is False:
            self.features = nn.Sequential(
                # 3*3卷积核，步长为1，填充1，不变图片尺寸
                nn.Conv2D(in_channels=num_channels,
                                out_channels=num_filters,
                                kernel_size = 3, 
                                stride=1, 
                                padding=1),
                nn.BatchNorm(num_filters,act="relu"),
                nn.Conv2D(in_channels=num_filters, 
                                out_channels=num_filters,
                                kernel_size=3,
                                stride=1,
                                padding=1),             
                nn.BatchNorm(num_filters, act="relu")
            )
        elif attention=='CBAM':
            self.features = nn.Sequential(
                # 3*3卷积核，步长为1，填充1，不变图片尺寸
                nn.Conv2D(in_channels=num_channels,
                                out_channels=num_filters,
                                kernel_size = 3, 
                                stride=1, 
                                padding=1),
                nn.BatchNorm(num_filters,act="relu"),
                nn.Conv2D(in_channels=num_filters, 
                                out_channels=num_filters,
                                kernel_size=3,
                                stride=1,
                                padding=1),
                CBAM(num_filters),             
                nn.BatchNorm(num_filters, act="relu")
                )
        elif attention=='SEAttention':
            self.features = nn.Sequential(
                # 3*3卷积核，步长为1，填充1，不变图片尺寸
                nn.Conv2D(in_channels=num_channels,
                                out_channels=num_filters,
                                kernel_size = 3, 
                                stride=1, 
                                padding=1),
                nn.BatchNorm(num_filters,act="relu"),
                nn.Conv2D(in_channels=num_filters, 
                                out_channels=num_filters,
                                kernel_size=3,
                                stride=1,
                                padding=1),
                SEAttention(num_filters),             
                nn.BatchNorm(num_filters, act="relu")
                )

        # 池化层，图片尺寸减半[H/2 W/2]
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding="SAME")
    def forward(self,inputs):
        x = self.features(inputs)
        x_conv = x
        x_pool = self.pool(x)
        # 返回池化之前的与  池化之后的
        return x_conv, x_pool 

# paddle.summary(Encoder(3,64,'SEAttention'), (1, 3, 128, 128))


# In[26]:


class Decoder(nn.Layer):
    def __init__(self, num_channels,num_filters,attention=False):
        super(Decoder,self).__init__()
        if attention is False:
            self.features = nn.Sequential(
                # 3*3卷积核，步长为1，填充1，不变图片尺寸
                nn.Conv2D(in_channels=num_channels,
                                out_channels=num_filters,
                                kernel_size = 3, 
                                stride=1, 
                                padding=1),
                nn.BatchNorm(num_filters,act="relu"),
                nn.Conv2D(in_channels=num_filters, 
                                out_channels=num_filters,
                                kernel_size=3,
                                stride=1,
                                padding=1),             
                nn.BatchNorm(num_filters, act="relu")
            )

        elif attention=='CBAM':
            self.features = nn.Sequential(
                # 3*3卷积核，步长为1，填充1，不变图片尺寸
                nn.Conv2D(in_channels=num_channels,
                                out_channels=num_filters,
                                kernel_size = 3, 
                                stride=1, 
                                padding=1),
                nn.BatchNorm(num_filters,act="relu"),
                nn.Conv2D(in_channels=num_filters, 
                                out_channels=num_filters,
                                kernel_size=3,
                                stride=1,
                                padding=1),
                CBAM(num_filters),             
                nn.BatchNorm(num_filters, act="relu")
                )
        elif attention=='SEAttention':
            self.features = nn.Sequential(
                # 3*3卷积核，步长为1，填充1，不变图片尺寸
                nn.Conv2D(in_channels=num_channels,
                                out_channels=num_filters,
                                kernel_size = 3, 
                                stride=1, 
                                padding=1),
                nn.BatchNorm(num_filters,act="relu"),
                nn.Conv2D(in_channels=num_filters, 
                                out_channels=num_filters,
                                kernel_size=3,
                                stride=1,
                                padding=1),
                SEAttention(num_filters),             
                nn.BatchNorm(num_filters, act="relu")
                ) 
        # 图片尺寸变大一倍[2*H 2*W]
        self.up = nn.Conv2DTranspose(num_channels,num_filters,2,2,padding=0)

    def forward(self,input_conv,input_pool):
        
        x = self.up(input_pool)
        h_diff = (input_conv.shape[2] - x.shape[2])
        w_diff = (input_conv.shape[3] - x.shape[3])
        # #以下采样保存的feature map为基准，填充上采样的feature map尺寸
        # padding=[上，下，左，右]填充空白像素
        pad = nn.Pad2D(padding=[h_diff//2, h_diff - h_diff//2, w_diff//2, w_diff-w_diff//2])
        x = pad(x)
        # in_channels扩大两倍
        x = paddle.concat([input_conv, x], axis=1)
        x = self.features(x)
        return x


# In[27]:


class UNet(nn.Layer):
    def __init__(self,num_classes=2):
        super(UNet,self).__init__()
        self.down1 = Encoder(num_channels=  3, num_filters=64,attention=False) #下采样
        self.down2 = Encoder(num_channels= 64, num_filters=128,attention=False)
        self.down3 = Encoder(num_channels=128, num_filters=256,attention=False)
        self.down4 = Encoder(num_channels=256, num_filters=512,attention=False)
        
        self.mid_conv1 = nn.Conv2D(512,1024,1)                 #中间层
        self.mid_bn1   = nn.BatchNorm(1024,act="relu")
        self.mid_conv2 = nn.Conv2D(1024,1024,1)
        self.mid_bn2   = nn.BatchNorm(1024,act="relu")

        self.up4 = Decoder(1024,512,attention=False)                           #上采样
        self.up3 = Decoder(512,256,attention=False)
        self.up2 = Decoder(256,128,attention=False)
        self.up1 = Decoder(128,64,attention=False)
        
        self.last_conv = nn.Conv2D(64,num_classes,1)           #1x1卷积，softmax做分类
        
    def forward(self,inputs):
        x1, x = self.down1(inputs)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        
        x = self.last_conv(x)
        
        return x


# paddle.summary(UNet(2), (1, 3, 128, 128))


# In[28]:


class SEAttention_UNet(nn.Layer):
    def __init__(self,num_classes=2):
        super(SEAttention_UNet,self).__init__()
        self.down1 = Encoder(num_channels=  3, num_filters=64, attention='SEAttention') #下采样
        self.down2 = Encoder(num_channels= 64, num_filters=128,attention='SEAttention')
        self.down3 = Encoder(num_channels=128, num_filters=256,attention='SEAttention')
        self.down4 = Encoder(num_channels=256, num_filters=512,attention='SEAttention')

        self.mid_conv1 = nn.Conv2D(512,1024,1)                 #中间层
        self.mid_bn1   = nn.BatchNorm(1024,act="relu")
        self.mid_conv2 = nn.Conv2D(1024,1024,1)
        self.mid_bn2   = nn.BatchNorm(1024,act="relu")

        self.up4 = Decoder(1024,512,attention=False)                           #上采样
        self.up3 = Decoder(512,256,attention=False)
        self.up2 = Decoder(256,128,attention=False)
        self.up1 = Decoder(128,64,attention=False)
        
        self.last_conv = nn.Conv2D(64,num_classes,1)           #1x1卷积，softmax做分类
        
    def forward(self,inputs):
        x1, x = self.down1(inputs)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        x = self.last_conv(x)
        logit_list = [x]
        # print("Output shape:", x.shape)
        return logit_list


# paddle.summary(SEAttention_UNet(), (1, 3, 128, 128))


# In[29]:


class CBAM_UNet(nn.Layer):
    def __init__(self,num_classes=2):
        super(CBAM_UNet,self).__init__()
        self.down1 = Encoder(num_channels=  3, num_filters=64, attention='CBAM') #下采样
        self.down2 = Encoder(num_channels= 64, num_filters=128,attention='CBAM')
        self.down3 = Encoder(num_channels=128, num_filters=256,attention='CBAM')
        self.down4 = Encoder(num_channels=256, num_filters=512,attention='CBAM')
        
        
        self.mid_conv1 = nn.Conv2D(512,1024,1)                 #中间层
        self.mid_bn1   = nn.BatchNorm(1024,act="relu")
        self.mid_conv2 = nn.Conv2D(1024,1024,1)
        self.mid_bn2   = nn.BatchNorm(1024,act="relu")

        self.up4 = Decoder(1024,512,attention=False)                           #上采样
        self.up3 = Decoder(512,256,attention=False)
        self.up2 = Decoder(256,128,attention=False)
        self.up1 = Decoder(128,64,attention=False)
        
        self.last_conv = nn.Conv2D(64,num_classes,1)           #1x1卷积，softmax做分类
        
    def forward(self,inputs):
        x1, x = self.down1(inputs)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        
        x = self.last_conv(x)
        logit_list = [x]
        
        return logit_list


# paddle.summary(CBAM_UNet(), (1, 3, 128, 128))


# # **SE_CBAM_UNet训练还是不行**

# In[30]:


import paddle
import paddle.nn as nn


class SE_CBAM_UNet(nn.Layer):
    def __init__(self, num_classes=2):
        super(SE_CBAM_UNet, self).__init__()

        self.down1 = Encoder(num_channels=3, num_filters=64, attention='SEAttention')
        self.down2 = Encoder(num_channels=64, num_filters=128, attention='SEAttention')
        self.down3 = Encoder(num_channels=128, num_filters=256, attention='SEAttention')
        self.down4 = Encoder(num_channels=256, num_filters=512, attention='SEAttention')

  
        self.mid_conv1 = nn.Conv2D(512, 1024, 1)
        self.mid_bn1 = nn.BatchNorm2D(1024)
        self.mid_conv2 = nn.Conv2D(1024, 1024, 1)
        self.mid_bn2 = nn.BatchNorm2D(1024)

        self.up4 = Decoder(1024, 512, attention='CBAM')
        self.up3 = Decoder(512, 256, attention='CBAM')
        self.up2 = Decoder(256, 128, attention='CBAM')
        self.up1 = Decoder(128, 64, attention=False)  

   
        self.final_conv = nn.Conv2D(64, num_classes, 1)

    def forward(self, x):
        
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

       
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

       
        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)

       
        output = self.final_conv(x)
        return [output]


# # **SEAttention**

# In[ ]:


# 带有注意力机制的模型已经建立完成
# 现在开始训练
paddle.device.set_device('gpu:0')
# from paddle.nn import CrossEntropyLoss

# losses = {
#     'types': [CrossEntropyLoss()],
#     'coef': [1]
# }

losses = {}
# losses['types'] = [LovaszSoftmaxLoss()]
# losses['coef'] = [1]
losses['types'] = [BCELoss()] 
losses['coef'] = [1]
batch_size = 16
seunet_model = SEAttention_UNet()

# 为seunet_model创建优化器
seunet_optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-3,
    parameters=seunet_model.parameters(),
    weight_decay=4.0e-5
)

# 训练attention_unet
train(
    model=seunet_model,
    filename='seunet_model.npy',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=seunet_optimizer,
    save_dir='output_seunet',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/5),
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True
)


# # CBAM

# In[ ]:


# 带有注意力机制的模型已经建立完成
# 现在开始训练
paddle.device.set_device('gpu:0')
# from paddle.nn import CrossEntropyLoss

# losses = {
#     'types': [CrossEntropyLoss()],
#     'coef': [1]
# }

losses = {}
# losses['types'] = [LovaszSoftmaxLoss()]
# losses['coef'] = [1]
losses['types'] = [BCELoss()] 
losses['coef'] = [1]
batch_size = 16
seunet_model = CBAM_UNet()

# 为seunet_model创建优化器
seunet_optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-3,
    parameters=seunet_model.parameters(),
    weight_decay=4.0e-5
)

# 训练attention_unet
train(
    model=seunet_model,
    filename='CBAMunet_model.npy',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=seunet_optimizer,
    save_dir='output_CNAMunet',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/5),
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True
)


# # **# SEAttention+CBAM还是有问题**
# ValueError: (InvalidArgument) Input(X) and Input(Label) shall have the same rank.But received: the rank of Input(X) is [4], the rank of Input(Label) is [3].
#   [Hint: Expected rank == labels_dims.size(), but received rank:4 != labels_dims.size():3.] (at /paddle/paddle/fluid/operators/bce_loss_op.cc:43)
#   [operator < bce_loss > error]

# In[31]:


# 带有注意力机制的模型已经建立完成
# 现在开始训练
paddle.device.set_device('gpu:0')
# from paddle.nn import CrossEntropyLoss

# losses = {
#     'types': [CrossEntropyLoss()],
#     'coef': [1]
# }

losses = {}
# losses['types'] = [LovaszSoftmaxLoss()]
# losses['coef'] = [1]
losses['types'] = [BCELoss()] 
losses['coef'] = [1]
batch_size = 16
seunet_model = SE_CBAM_UNet()

# 为seunet_model创建优化器
seunet_optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-3,
    parameters=seunet_model.parameters(),
    weight_decay=4.0e-5
)

# 训练attention_unet
train(
    model=seunet_model,
    filename='SE_CBAM_UNet_model.npy',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=seunet_optimizer,
    save_dir='output_SE_CBAM_UNet',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/5),
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True
)


# In[ ]:


# 读取图像绘图 
# 可以加一些修饰
loss=np.load('output_unet/unet.npy')
x=range(1,len(loss)+1)
plt.plot(x,loss)


# In[ ]:


import os
def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir
get_ipython().system('pwd')
image_path = 'dataset/test1_img/' # 也可以输入一个包含图像的目录
image_list, image_dir = get_image_list('dataset/test1_img/')


# In[ ]:


get_ipython().system('pwd')
from paddleseg.utils import get_sys_env, logger, get_image_list
# image_path='dataset/test2_img/'
# image_list, image_dir = get_image_list(image_path)

import paddleseg.transforms as T
test_transforms = T.Compose([
    T.Resize(target_size=(256, 256)),
    T.Normalize()
])
from paddleseg.core import predict
predict(
        model=unet_model,
        model_path='output_unet/best_model/model.pdparams',
        transforms=test_transforms,
        image_list=image_list,
        image_dir='dataset/test1_img',
        save_dir='output_unet/results1'
    )


# In[ ]:


def evaulate_miou(img_path,label_path):
    # print(imgs)
    iou=[]
    for img in os.listdir(img_path):
        imgpath=img_path+img
        img_1=cv2.imread(imgpath)
        img_test=np.zeros(img_1.shape[0:2])
        for n,i in enumerate(img_1):
            for m,j in enumerate(i):
                if j[1]==128:
                    img_test[n][m]=1
        data_path=label_path+img
        data=cv2.imread(data_path,-1)

        target=data
        prediction=img_test
        intersection = np.logical_and(target, prediction) 
        union = np.logical_or(target, prediction) 
        iou_score = np.sum(intersection) / np.sum(union)
        iou.append(iou_score)
    return iou


# In[ ]:


img_path='/home/aistudio/PaddleSeg/output_unet/results1/pseudo_color_prediction/'
label_path='/home/aistudio/PaddleSeg/dataset/test1_label_img_in/'
iou_unet=evaulate_miou(img_path,label_path)
print(len(iou_unet))
print(sum(iou_unet)/len(iou_unet))
x=range(0,len(iou_unet))
y=iou_unet
plt.plot(x,y)
plt.show()


# In[ ]:



# 绘制箱型图，观察离群值，说明大部分是正常预测的，除了极端部分，再解释下极端部分即可
fig, ax = plt.subplots()      # 子图
data=iou_unet
ax.boxplot(data)


# In[ ]:





# In[ ]:




