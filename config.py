import os
import torch

class Config:
    """
    配置类，用于管理所有训练和模型参数
    """
    def __init__(self):
        # 数据路径
        self.train_images = os.path.join('data', 'train', 'images')
        self.train_labels = os.path.join('data', 'train', 'labels')
        self.val_images = os.path.join('data', 'val', 'images')
        self.val_labels = os.path.join('data', 'val', 'labels')
        
        # 模型参数
        self.num_classes = 3  # 分割类别数
        
        # 训练参数
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        
        # 损失函数权重
        self.weight_ce = 0.5  # 交叉熵权重
        self.weight_dice = 0.5  # Dice损失权重
        
        # 日志和保存
        self.log_interval = 10  # 每隔多少batch打印一次日志
        self.model_save_path = os.path.join('models', 'unet_resnet.pth')
        
        # 分布式训练
        self.world_size = torch.cuda.device_count()
        
        # 数据增强参数
        self.rotation_range = 15  # 随机旋转角度范围
        self.scale_range = (0.8, 1.2)  # 随机缩放范围
