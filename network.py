import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)# torch.Size([1, 512, 18, 32])
        x = self.up_16_8(f8, x)# torch.Size([1, 256, 36, 64])
        x = self.up_8_4(f4, x)# torch.Size([1, 256, 72, 128])
        x = self.pred(F.relu(x))# torch.Size([1, 1, 72, 128])        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)# torch.Size([1, 1, 288, 512])
        return x



#space-time memory read
class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, mk, mv, qk, qv):#kM,vM, kQ,vQ
        B, CK, T, H, W = mk.shape
        _, CV, _, _, _ = mv.shape

        mi = mk.view(B, CK, T*H*W) #将时间和空间维度合并。
        mi = torch.transpose(mi, 1, 2) # B * THW * CK
        #
        #除以 math.sqrt(CK) 进行归一化，有助于稳定学习过程
        qi = qk.view(B, CK, H*W) / math.sqrt(CK)  # B * CK * HW

        # 计算输入特征与查询特征之间的亲和度,相似度
        affinity = torch.bmm(mi, qi)  # B, THW, HW
        affinity = F.softmax(affinity, dim=1)  # B, THW, HW

        # 根据亲和度对记忆特征进行加权求和
        mv = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mv, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        # 将记忆特征与查询特征合并
        mem_out = torch.cat([mem, qv], dim=1)
        
        #mem_out：包含从记忆中检索到的信息和当前查询信息的融合特征图
        return mem_out


class PropagationNetwork(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        if single_object:
            self.mask_rgb_encoder = MaskRGBEncoderSO() 
        else:
            self.mask_rgb_encoder = MaskRGBEncoder() 
        self.rgb_encoder = RGBEncoder()

        self.kv_m_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.kv_q_f16 = KeyValue(1024, keydim=128, valdim=512)

        self.memory = MemoryReader()
        self.decoder = Decoder()
        self.aspp = ASPP(1024)
        self.score = Score(1024)
        self.concat_conv = nn.Conv2d(1025, 1, 3, 1, 1)

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def memorize(self, frame, mask, other_mask=None): 
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.mask_rgb_encoder(frame, mask)#torch.Size([1, 1024, 18, 32])

     
        else:
            f16 = self.mask_rgb_encoder(frame, mask, other_mask)
        k16, v16 = self.kv_m_f16(f16)#torch.Size([1, 128, 18, 32]) torch.Size([1, 512, 18, 32])

        mask_score = self.score(f16)#torch.Size([1, 1])

        return k16.unsqueeze(2), v16.unsqueeze(2), mask_score #bchw-> B*C*T*H*W

    def segment(self, frame, keys, values, mask1=None, mask2=None, selector=None): 
        """
        对给定的视频帧进行分割，根据提供的键（keys）和值（values）以及可选的掩码和选择器来更新记忆模块，并产生分割的logits和概率。

        参数:
        - frame: 输入的视频帧，用于通过RGB编码器增强分割过程。
        - keys: 键张量，用于记忆模块的更新。
        - values: 值张量，与键对应，用于记忆模块的更新。
        - mask1: 可选，第一个掩码，用于单对象或多对象的场景中。
        - mask2: 可选，第二个掩码，仅用于多对象场景中。
        - selector: 可选，选择器张量，用于在多对象场景中加权概率。

        返回值:
        - logits: 分割逻辑的张量，用于表示每个像素属于前景的概率。
        - prob: 分割概率的张量，表示每个像素属于前景的概率。
        """
        


        b, k = keys.shape[:2]# 1 128
        ###enhance
        if self.single_object:
            mask = mask1.clone().detach()
        else:
            mask1_detach = mask1.clone().detach()
            mask2_detach = mask2.clone().detach()
            mask1_detach = mask1_detach.unsqueeze(0)
            mask2_detach = mask2_detach.unsqueeze(0)
            mask_all = torch.cat([mask1_detach, mask2_detach], dim=0)
            mask, _ = torch.max(mask_all, dim=0)
            
                
 

        # 通过RGB编码器获取不同尺度的特征
        f16, f8, f4 = self.rgb_encoder(frame)
        b, c, h, w = f16.size()
        # 使用双线性插值将掩码重塑以匹配特征的尺寸，并与f16特征合并
        mask_reshape = F.interpolate(mask, size=[h, w], mode='bilinear')
        concat_f16 = torch.cat([f16, mask_reshape], dim=1)      #B,C+1,H,W  torch.Size([1, 1024, 18, 32]) torch.Size([1, 1, 18, 32]) torch.Size([1, 1025, 18, 32]) 
        concat_f16 = torch.sigmoid(self.concat_conv(concat_f16))# torch.Size([1, 1, 18, 32])

        concat_f16 = f16 * concat_f16 # torch.Size([1, 1024, 18, 32])
        

        # 使用concat_f16计算键和值    kQ,vQ
        k16, v16 = self.kv_q_f16(concat_f16)        #B,C,H,W
        
        if self.single_object:

            mr = self.memory(keys, values, k16, v16)#kM,vM, kQ,vQ#torch.Size([1, 1024, 18, 32]) 从记忆帧计算当前帧得来的
            mr = self.aspp(mr)#torch.Size([1, 1024, 18, 32])

            
            logits = self.decoder(mr, f8, f4)# torch.Size([1, 1, 288, 512])

            prob = torch.sigmoid(logits)#torch.Size([1, 1, 288, 512])

            
      

        else:
            mr_0 = self.memory(keys[:,0], values[:,0], k16, v16)
            mr_0 = self.aspp(mr_0)
            logits_0 = self.decoder(mr_0, f8, f4)
            mr_1 = self.memory(keys[:,1], values[:,1], k16, v16)
            mr_1 = self.aspp(mr_1)
            logits_1 = self.decoder(mr_1, f8, f4)
            logits = torch.cat([logits_0, logits_1], dim=1)
            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)



        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]


        return logits, prob
    '''logits 通常指的是一个模型输出层的原始预测值，这些值还没有通过激活函数（如 softmax 或 sigmoid）转换成概率。
    在这个上下文中: logits 是从解码器（self.decoder）输出的，
    随后通过 aggregate 函数进一步处理的值。这些原始预测值经过处理后用于计算每个像素点属于前景（目标对象）的概率。
    处理: logits 首先被 aggregate 函数处理，该函数使用对数几率转换将原始输出转换为更稳定的形式，以进行下一步的概率计算。

    prob
    定义: prob 代表概率，即模型预测每个像素点属于各个类别的概率。
    在这个上下文中: prob 是从 logits 转换得来的，
    使用 Sigmoid 或 Softmax 函数将 logits 转换成概率值。
    具体使用哪种函数取决于模型的设计和任务的需求（本代码使用了 Sigmoid 函数后面还结合了 Softmax 函数进行处理）。
    处理:
    如果是单对象模式，直接应用 Sigmoid 函数计算概率。
    如果是多对象模式，计算每个对象的预测后，通过 Sigmoid 函数转换，
    并使用 selector 来加权这些概率（selector 可能用于选择哪个对象的预测更可靠或适用于当前帧）。
    然后使用 Softmax 函数标准化这些概率，使其总和为 1。
    
    
    logits 是模型对每个像素点分类的原始预测，而 prob 是这些预测被转换成的概率形式，
    表明每个像素属于前景的概率。这些概率可以用于生成最终的分割掩码，用于视频中对象的视觉跟踪和识别。'''
    def forward(self, *args, **kwargs):
        
        if args[1].dim() > 4: # keys

            return self.segment(*args, **kwargs)
        else:

            return self.memorize(*args, **kwargs)


