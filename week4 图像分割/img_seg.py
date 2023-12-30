#from paddleseg.models import unet
#导入UNET模型
# !pip install paddleseg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2 
import paddle
import os
#from visualdl import LogWriter
from paddleseg.models import BiSeNetV2,UNet
from paddleseg.datasets import OpticDiscSeg
from paddleseg.core import evaluate
from paddleseg.models.losses import CrossEntropyLoss
import paddleseg.transforms as T
from paddleseg.core import train
from paddleseg.core import predict

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


model = UNet(num_classes=2,
                 align_corners=False,
                 pretrained=None)


# 构建训练用的数据增强和预处理
transforms = [
    T.Resize(target_size=(512, 512)),
    T.RandomHorizontalFlip(),
    T.Normalize()
]

# 构建训练集
from paddleseg.datasets import OpticDiscSeg
train_dataset = OpticDiscSeg(
    dataset_root='data/optic_disc_seg',
    transforms=transforms,
    mode='train'
)

# 构建验证用的数据增强和预处理
transforms = [
    T.Resize(target_size=(512, 512)),
    T.Normalize()
]

# 构建验证集
val_dataset = OpticDiscSeg(
    dataset_root='data/optic_disc_seg',
    transforms=transforms,
    mode='val'
)

# 设置学习率
base_lr = 0.01
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=1000, end_lr=0)

optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)

#构建损失函数
losses = {}
losses['types'] = [CrossEntropyLoss()] * 1
losses['coef'] = [1]* 1

#模型训练
train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output',
    iters=1000,
    batch_size=4,
    save_interval=200,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)


#加载训练后的模型
model_path = 'output/best_model/model.pdparams'
if model_path:
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    print('Loaded trained params of model successfully')
else: 
    raise ValueError('The model_path is wrong: {}'.format(model_path))



#可视化
transforms = T.Compose([
    T.Resize(target_size=(512, 512)),
    T.RandomHorizontalFlip(),
    T.Normalize()
])
# 读取test_list.txt文件
test_list_path = 'data/optic_disc_seg/test_list.txt'
with open(test_list_path, 'r') as f:
    lines = f.readlines()
img_list=[]
label_list=[]
# 循环遍历每一行并加载图像和标注
for line in lines:
    image_path, annotation_path = line.strip().split()  # 分割每行以获取图像和标注路径\
    image_path = os.path.join('data/optic_disc_seg/', image_path)
    annotation_path = os.path.join('data/optic_disc_seg/', annotation_path)
    #image_path = os.path.join(current_directory, 'data', 'optic_disc_seg', 'test_list.txt')
    img_list.append(image_path)
    label_list.append(annotation_path)


for j in range(len(img_list)):
    image_list, image_dir = get_image_list(img_list[j])
    #print(image_list,image_dir)
    predict(
            model,
            model_path='output/best_model/model.pdparams',
            transforms=transforms,
            image_list=image_list,
            image_dir=image_dir,
            save_dir='output/results/case{}'.format(j)
        )
    mask_img = os.path.join('output/results/case{}'.format(j),'added_prediction')
    image_files = os.listdir(mask_img)
    image_file = image_files[0]
    img_mask = os.path.join(mask_img, image_file)
    img_dir = [img_list[j], label_list[j],img_mask]
    plt.figure(figsize=(15, 15))

    title = ['Input_Image','label_image','Predicted Mask']
        
    for i in range(len(title)):
        plt.subplot(1, len(title), i+1)
        plt.title(title[i])
        img = plt.imread(img_dir[i])
        plt.imshow(img)
        plt.axis('off')
        plt.savefig('./result{}'.format(j))
    # plt.show()


# 构建验证用的transforms
transforms = [
    #T.Resize(target_size=(512, 512)),
    T.Normalize()
]

# 构建测试集测试
test_dataset = OpticDiscSeg(
    dataset_root='data/optic_disc_seg',
    transforms=transforms,
    mode='test'
)
print("正常评估:")
evaluate(
        model,
        test_dataset)
#多尺度+翻转评估
print("多尺度+翻转评估:")
evaluate(
        model,
        test_dataset,
        aug_eval=True,
        scales=[0.75, 1.0, 1.25],
        flip_horizontal=True)

