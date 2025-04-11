import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torchvision.models import resnet50,ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch.distributed as dist

class MyDataset(nn.Module):
    def __init__(self,data_dir=r"data/VOCdevkit/VOC2012",image_h=380,image_w=380,is_train=True,trans=None):
        super().__init__()
        self.h,self.w=image_h,image_w
        self.is_train=is_train
        self.images,self.masks=self.read_from_dir(data_dir)
        self.transform=trans
        if self.transform==None:
            if self.is_train:
                self.aug=A.Compose([
                A.Resize(self.h,self.w),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                A.RandomBrightnessContrast(
                brightness_limit=0.2,contrast_limit=0.2,p=0.5),
                A.Blur()],p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
                ToTensorV2()
                ])
            else:
                self.aug=A.Compose([
                A.Resize(self.h,self.w),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
                ToTensorV2()
                ])
        else:
            self.aug=self.transform
        
    def __getitem__(self,index):
        image=self.images[index]
        mask=self.masks[index]
        image_pil=Image.open(image)
        mask_pil=Image.open(mask)
        image_np=np.array(image_pil)
        mask_np=np.array(mask_pil).astype(np.int64)
        augmented=self.aug(image=image_np,mask=mask_np)
        image=augmented['image']
        mask=augmented['mask']
        mask[mask==255]=-1
        return image,mask
    
    def __len__(self):
        return len(self.images)
        
    def read_from_dir(self,data_dir):
        if self.is_train:
            img_dir=os.path.join(data_dir,"ImageSets","Segmentation","train.txt")
        else:
            img_dir=os.path.join(data_dir,"ImageSets","Segmentation","val.txt")
        file=open(img_dir)
        images=[]
        masks=[]
        for line in file:
            line=line.strip("\n")
            images.append(os.path.join(data_dir,"JPEGImages",line+".jpg"))
            masks.append(os.path.join(data_dir,"SegmentationClass",line+".png"))
        return images,masks

val_ds=DataLoader(MyDataset(is_train=False),batch_size=15,shuffle=False,num_workers=0)

def setup():
    init_process_group("nccl")

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2,ignore_dix=-1, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.alpha=alpha
        self.gamma=gamma
        self.ignore_dix=ignore_dix
    def forward(self,inputs,targets):
        predict=inputs.permute(0,2,3,1).contiguous()
        predict=torch.softmax(predict,dim=-1)
        b,c=predict.size(0),predict.size(3)
        mask=targets!=self.ignore_dix
        predict=predict[mask].view(-1,c)
        targets=targets[mask].view(-1)
        one_hot=torch.eye(c,device=predict.device)
        targets=one_hot[targets].view(-1,c).float()
        FL=((-self.alpha*((1-predict)**self.gamma))*targets*torch.log2(predict+1e-12)).sum(dim=-1)
        return FL.mean()
    
## 定义累加器
class Accumulator():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self, item):
        return self.data[item]
    
def miou(pred,target,num_classes=21):
    cm=np.zeros((num_classes,num_classes),dtype=np.int64)
    pred=pred.argmax(dim=1)
    y_pred_flat=pred.detach().cpu().numpy().flatten()
    y_true_flat=target.detach().cpu().numpy().flatten()
    for t,p in zip(y_true_flat,y_pred_flat):
        cm[t][p]+=1
    sum_true=np.sum(cm,axis=1)
    sum_pred=np.sum(cm,axis=0)
    tp=np.diag(cm)
    denominator=sum_true+sum_pred-tp
    iou=tp/denominator
    iou[denominator==0]=np.nan
    return np.nanmean(iou)

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
    
class PPM(nn.Module):
    def __init__(self,in_channels,bin_sizes=[1,2,3,6]):
        super(PPM,self).__init__()
        out_channels=int(in_channels//len(bin_sizes))
        self.stages=nn.ModuleList()
        for bin_size in bin_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                ConvBlock(in_channels,out_channels,1,1,0),
            ))
    def forward(self,x):
        _,_,h,w=x.size()
        out=[x]
        for stage in self.stages:
            out.append(F.interpolate(stage(x),size=(h,w),mode='bilinear',align_corners=True))
        output=torch.cat(out,dim=1)
        return output
    
class PSPNet(nn.Module):
    def __init__(self,num_classes=21):
        super(PSPNet,self).__init__()
        self.C=num_classes
        self.backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.set_dilation(self.backbone)
        self.head=ConvBlock(3,64,kernel_size=3,stride=2,padding=1)
        self.layer1=self.backbone.layer1
        self.layer2=self.backbone.layer2
        self.layer3=self.backbone.layer3
        self.layer4=self.backbone.layer4

        self.ppm=PPM(2048,bin_sizes=[1,2,3,6])

        self.main_conv=nn.Sequential(ConvBlock(4096,512,kernel_size=3,stride=1,padding=1),
                                  ConvBlock(512,256,kernel_size=3,stride=1,padding=1),
                                  ConvBlock(256,self.C,kernel_size=1,stride=1,padding=0))
        self.aux_conv=nn.Sequential(ConvBlock(1024,256,kernel_size=3,stride=1,padding=1),
                          ConvBlock(256,self.C,kernel_size=1,stride=1,padding=0))
    def set_dilation(self,layer,d=2):
        p=d
        s=1
        for n,m in layer.named_modules():
            if n in ["layer3.1.conv2","layer3.2.conv2","layer3.3.conv2","layer3.4.conv2","layer3.5.conv2",
                     "layer.4.0.conv2"]:
                m.dilation=(d,d)
                m.padding=(p,p)
            elif n in ["layer3.0.conv2"]:
                m.stride=(s,s)
            elif n in ["layer3.0.downsample.0"]:
                m.stride=(s,s)
            elif n in ["layer4.1.conv2","layer4.2.conv2"]:
                m.dilation=(2*d,2*d)
                m.padding=(2*p,2*p)
    def forward(self,x):
        _,_,h,w=x.size()
        x=self.head(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x_aux=self.layer3(x)
        x=self.layer4(x_aux)

        x=self.ppm(x)
        x=self.main_conv(x)
        x=F.interpolate(x,size=(h,w),mode='bilinear',align_corners=True)
        aux=self.aux_conv(x_aux)
        aux=F.interpolate(aux,size=(h,w),mode='bilinear',align_corners=True)
        return x,aux

def train(model,epoches,train_dataset,lr,optimizer=None,free=False,rank=int(os.environ["LOCAL_RANK"])):
    print("training...")
    history=[]
    epoch_run=0
    model.to(rank)
    model=DDP(model,device_ids=[rank],find_unused_parameters=True)
    ## load model
    if os.path.exists("pspnet.pth"):
        checkpoint = torch.load("pspnet.pth", map_location=f"cuda:{rank}",weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", {}))
        epoch_run = checkpoint.get("epoch", 0)
    else:
        checkpoint = {}
    ## freeze layers
    for name,param in model.named_parameters():
        if "layer1" in name or "layer2" in name or "layer3" in name or "layer4" in name:
            param.requires_grad = free
    ## create loader
    train_sampler=DistributedSampler(train_dataset,shuffle=True)
    train_loader=DataLoader(train_dataset,batch_size=4,sampler=train_sampler,num_workers=0,pin_memory=True,shuffle=False)
    ## load optimizer
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4) if optimizer==None else optimizer
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss_fn=FocalLoss(ignore_dix=-1)
    ## train
    for epoch in range(epoch_run,epoches):
        model.train()
        accum=Accumulator(2)
        for i,(X,y) in enumerate(train_loader):
            X,y=X.to(rank),y.to(rank)
            optimizer.zero_grad()
            pred,aux=model(X)
            loss=loss_fn(pred,y)+0.4*loss_fn(aux,y)
            loss.sum().backward()
            optimizer.step()
            accum.add(loss.item(),1)
        history.append(accum[0]/accum[1])
        print(f"\tepoch {epoch+1} train loss: {accum[0]/accum[1]:.3e}")
        ## save model
        if (epoch+1)%5==0:
            if rank==0:
                ckp = {
        "model_state_dict": model.state_dict(),  # 关键：使用 model.module 获取原始模型
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,}
                torch.save(ckp,f"pspnet.pth")

    if rank==0:
        plt.plot(history)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("train.png")

def main(model,train_dataset,lr,epoches,optimizer=None,free=False,rank=int(os.environ["LOCAL_RANK"])):
    setup()
    train(model,epoches,train_dataset,lr,optimizer,free,rank)
    destroy_process_group()

if __name__ == "__main__":
    "torchrun --standalone --nproc_per_node=gpu multi_gpu_pspnet.py"
    model,train_dataset,val_dataset,lr,epoches,optimizer,free,rank=PSPNet(),MyDataset(is_train=True),MyDataset(is_train=False),1e-6,405,None,True,int(os.environ["LOCAL_RANK"])
    main(model,train_dataset,lr,epoches,optimizer,free,rank)