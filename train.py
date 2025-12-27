import warnings, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"    
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    
    model = RTDETR('E:\论文实验\Epro\RSD-DETR\ultralytics\cfg\models\model.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4, # 
                # device='0,1', #
                # resume='', # last.pt path
                project='runs/train',
                name='exp_EfficientVIM_SEContext_COK',
                )
    
