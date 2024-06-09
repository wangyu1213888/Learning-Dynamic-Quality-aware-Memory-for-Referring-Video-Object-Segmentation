"""
ReferFormer model class.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import os
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import CrossModalFPNDecoder, VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .propagator import VOSHead
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
import torch.utils.checkpoint as cp
import copy
from einops import rearrange, repeat

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class ReferFormer(nn.Module):
    """ This is the ReferFormer module that performs referring video object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, 
                    num_frames, mask_dim, dim_feedforward,
                    controller_layers, dynamic_mask_channels, 
                    aux_loss=False, with_box_refine=False, two_stage=False, 
                    freeze_text_encoder=False, rel_coord=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         ReferFormer can detect in a video. For ytvos, we recommend 5 queries for each frame.
            num_frames:  number of clip frames
            mask_dim: dynamic conv inter layer channel number.
            dim_feedforward: vision-language fusion module ffn channel number.
            dynamic_mask_channels: the mask feature output channel number.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        
        # Build Transformer
        # NOTE: different deformable detr, the query_embed out channels is
        # hidden_dim instead of hidden_dim * 2
        # This is because, the input to the decoder is text embedding feature
        self.query_embed = nn.Embedding(num_queries, hidden_dim) 
        
        # Build propagator
        #self.voshead = VOSHead(backbone.num_channels[:3]).eval()
        self.voshead = VOSHead(backbone.num_channels[:3])
                # 使用 1x1 卷积减少通道数
        self.conv_refer = torch.nn.Conv2d(2, 1, 1)  # 输入通道为 2，输出通道为 1，卷积核大小为 1




        # follow deformable-detr, we use the last three stages of backbone
        #这段代码在 ReferFormer 类中负责设置并初始化输入投影模块（input_proj），
        #用于将从主干网络（backbone）中提取的特征图（feature maps）
        #调整到与变换器（transformer）模块的隐藏维度（hidden_dim）匹配。
        #确保特征图能够有效地被后续的变换器网络处理。
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs): # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.mask_dim = mask_dim
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage == False, "args.two_stage must be false!"

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        #轮廓增强
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # Build Text Encoder
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # self.text_encoder = BertModel.from_pretrained('bert-base-cased')
        self.tokenizer = RobertaTokenizerFast.from_pretrained("./distilroberta_dow")
        self.text_encoder = RobertaModel.from_pretrained("./distilroberta_dow")  # distilroberta-base 'roberta-base'

        
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        
        # resize the bert output channel to transformer d_model
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        # Build FPN Decoder
        self.rel_coord = rel_coord##相对坐标
        #self.backbone.num_channels[0] 是主干网络（backbone）第一层的输出通道数，通常是高分辨率的特征图。
        #3 * [hidden_dim] 表示三层的输出特征通道数都设置为 hidden_dim，
        #这些通常是通过卷积层处理后得到的较低分辨率的特征图。这样的设计是为了确保FPN能够处理不同尺度的特征，以便检测不同大小的对象。
        feature_channels = [self.backbone.num_channels[0]] + 3 * [hidden_dim]
        self.pixel_decoder = CrossModalFPNDecoder(feature_channels=feature_channels, conv_dim=hidden_dim, 
                                                  mask_dim=mask_dim, dim_feedforward=dim_feedforward, norm="GN")

        # Build Dynamic Conv
        self.controller_layers = controller_layers #动态卷积模块的层数
        self.in_channels = mask_dim
        self.dynamic_mask_channels = dynamic_mask_channels#动态卷积的输出通道数，用于生成控制权重的特征图
        self.mask_out_stride = 4
        self.mask_feat_stride = 4

        #$循环计算每一层的权重数和偏置数。对于第一层，如果考虑相对坐标（rel_coord），则在输入通道数上加2（代表x和y坐标）。
        #中间层的权重和偏置数由动态掩码通道的平方和它本身决定。最后一层输出单个通道的权重和一个偏置。
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1) # output layer c -> 1
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        
        #多层感知机（MLP），其目的是根据输入特征动态生成卷积核的权重和偏置。
        #这使得模型能够针对不同的输入视频帧和区域生成特定的响应模式，从而提高对视频中动态对象的检测能力。
        #权重和偏置的初始化使用了Xavier均匀初始化方法和全零初始化，以保证训练的稳定性。
        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)   
        

    def forward(self, samples: NestedTensor, captions, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
               - captions: list[str]
               - targets:  list[dict]

            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]

               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples) 

        # features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        features, pos = self.backbone(samples) 

        b = len(captions)
        t = pos[0].shape[0] // b

        # For A2D-Sentences and JHMDB-Sentencs dataset, only one frame is annotated for a clip
        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            # t: num_frames -> 1
            t = 1


        #captions               list,
        #text_features          NestedTensor,
        #text_word_features     tensor
        text_features, text_sentence_features = self.forward_text(captions, device=pos[0].device)

        # prepare vision and text features for transformer
        srcs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose() 
        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]
        

        # res2:
        # 输出尺寸：56x56
        # 输出通道数：256 
        # res3:
        # 输出尺寸：28x28
        # 输出通道数：512
        # res4:
        # 输出尺寸：14x14
        # 输出通道数：1024
        # res5:
        # 输出尺寸：7x7
        # 输出通道数：2048
        # 1/4，1/8，1/16，1/32

        #最后生成的结果
        # ModuleList(
        #   (0): Sequential(
        #     (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        #     (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #   )
        #   (1): Sequential(
        #     (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        #     (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #   )
        #   (2): Sequential(
        #     (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        #     (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #   )
        #   (3): Sequential(
        #     (0): Conv2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #   )
        # )
        #最后的特征图富含信息： ResNet-50 的最后一个特征层（res5，通道数2048）包含了最丰富的语义信息，
        #它虽然在空间维度上较小，但在特征表达上最为丰富。使用它来生成额外的特征层可以保证这些高层特征被充分利用。
        # 维持特征深度： 通过对最后一个特征层进行进一步的下采样（如3x3卷积，步长2），
        #可以生成一个更小的特征图，同时保留了从原始特征图中学到的高级语义信息。这种方法可以在不丢失太多细节的情况下，提供一个尺度更大的特征表示。

        # Follow Deformable-DETR, we use the last three stages outputs from backbone
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])): #生成3层
            src, mask = feat.decompose()            
            src_proj_l = self.input_proj[l](src)    
            n, c, h, w = src_proj_l.shape

            # vision language early-fusion
            #改为由transformer融合
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            src_proj_l = self.fusion_module(tgt=src_proj_l,
                                             memory=text_word_features,
                                             memory_key_padding_mask=text_word_masks,
                                             pos=text_pos,
                                             query_pos=None
            ) 
            # src_proj_l = cp.checkpoint(self.fusion_module,src_proj_l,
            #                                  text_word_features,
            #                                  text_word_masks,
            #                                  text_pos,
            #                                  None)
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        #num_feature_levels定义为4
        if self.num_feature_levels > (len(features) - 1):#再生成一层
            _len_srcs = len(features) - 1 # fpn level _len_srcs=3
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])#下采样
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                # src = cp.checkpoint(self.fusion_module,src_proj_l,
                #                     text_word_features,
                #                     text_word_masks,
                #                     text_pos,
                #                     None)
                src = self.fusion_module(tgt=src,
                                          memory=text_word_features,
                                          memory_key_padding_mask=text_word_masks,
                                          pos=text_pos,
                                          query_pos=None
                )
                
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
        


        #经过 transformer 处理后，模型已经可以输出关于物体类别和边界框位置的预测值了。
        #再使用 pixel_decoder进行更精细的像素级别的预测，即对象分割，这通常涉及到每个像素点是否属于某一对象的分类。
        
        # num_queries是用于对象检测和分割任务中的一个关键参数。它决定了模型在每一帧中可以同时处理的最大对象数量。
        # num_queries的每个查询都是一个可学习的嵌入向量，这些向量作为模型预测的输入，帮助模型定位和识别图像或视频中的对象。
        # 对象检测：在对象检测任务中，模型会生成num_queries个对象预测，每个预测包含对象的类别和边界框。
        # 对象分割：在对象分割任务中，num_queries个查询会生成对应的分割掩码，标识图像中不同对象的区域。
        # 即使在只有一个对象需要分割的情况下，使用多个查询（num_queries）仍然可以增加模型的多样性和鲁棒性。
        # 多个查询可以捕捉到对象在不同尺度、位置和特征空间中的信息，从而提高分割的精确度和稳定性。例如，模型可能会生成多个重叠的分割结果，然后选择最优的结果进行输出
                
        # Transformer
        query_embeds = self.query_embed.weight  # [num_queries, c]
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
                                             self.transformer(srcs, text_embed, masks, poses, query_embeds)
        # hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
        #             cp.checkpoint(self._compute_transformer, srcs, text_embed, masks, poses, query_embeds)


        # hs (hidden states): 这是从transformer解码器得到的输出，其中包含了每个解码层的隐藏状态。这些隐藏状态后续会用来预测对象的类别、边界框等属性。
        #含义: hs 是解码器中各层的输出，包含了多层的隐藏状态。
        # 作用: 这些隐藏状态携带了解码过程中每一步的丰富信息。解码器使用来自编码器的特征（以及其他可能的输入，如位置编码或查询嵌入）来预测目标属性（如位置、类别等）。
        # 在模型中的应用:
        # 用于生成对象的细节特征，例如，它们可以直接用于预测对象的边界框、类别或者掩码。

        # hs 的每一层输出可以独立用于生成不同的预测，

        # 在训练过程中，它们也经常被用于计算辅助损失（auxiliary losses），帮助模型更好地学习。

        # memory: 这是变换器编码器的输出特征，用于存储视频帧的高维表示，这些特征将被解码器使用来进行对象查询和特征聚合。


        # 含义: memory_features 是编码器的输出，它被进一步处理（通常在特征金字塔网络中）用于匹配解码器使用的特征层级。
        # 作用: 这些特征是全局的、上下文化的图像表征，为解码器提供必要的背景信息以支持解码任务。
        # 在模型中的应用:
        # 在物体检测或分割任务中，memory_features 提供了高层次的语义信息，帮助模型识别和定位图像中的对象。
        # 由于它们经过编码器的多层处理，memory_features 包含了图像的深层次信息，这对于理解场景的复杂结构非常有用。
        # 它们通常被用作后续步骤（如特征金字塔网络或直接的对象分类和定位任务）的输入。


        # 关系：hs 和 memory_features 在模型中协同工作，memory_features 提供全局的、丰富的背景特征，
        # 而 hs 则是基于这些信息以及解码器自身的策略生成具体的预测输出。
        # 区别：hs 直接关联到最终的输出预测，如对象的类别和位置，而 memory_features 则更多地起到支持作用，为解码器提供了必要的上下文信息。
        
        # memory_features 是从编码器来的，提供了全局上下文信息，而 hs 是解码器的产物，直接用于生成针对每个目标的具体预测。


        # init_reference: 这是初始化的参考点，通常用于解码器的第一层来提供初始的空间参考，这有助于解码器确定对象的初步位置。
        # inter_references (intermediate references): 这是中间层的参考点，每个解码层都会输出新的参考点，这些参考点在迭代中逐步精细调整对象的位置。

        # 未使用两阶段模型
        # enc_outputs_coord_unact: 这是编码器输出的坐标，这些坐标未经激活，用于在解码器中提供初始的坐标参考。
        # enc_outputs_class: 如果是两阶段模型，这表示编码器输出的分类结果，用于第一阶段的提案分类。
        # enc_outputs_coord_unact: 如果是两阶段模型，这表示未激活的编码器输出坐标，这些坐标需要进一步处理以预测精确的对象边界框。
        # inter_samples: 这是解码器过程中采样的位置信息，有助于分析和调试模型的注意力聚焦和行为。


        #用torch.utils.checkpoint来解决显存不足问题
        #hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = checkpoint(self._compute_transformer, srcs, text_embed, masks, poses, query_embeds)



        # hs: [l, batch_size*time, num_queries_per_frame, c]
        # memory: list[Tensor], shape of tensor is [batch_size*time, c, hi, wi]
        # init_reference: [batch_size*time, num_queries_per_frame, 2]
        # inter_references: [l, batch_size*time, num_queries_per_frame, 4]
        
        out = {}
        # prediction
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):#hs.shape[0]->l is number of decoder layers
            # 这几行代码用于选择参考点，这些参考点用于定位目标的位置。对于第一层解码器，使用的是初始的参考点 (init_reference)，
            # 由解码器的第一层生成，基于对象查询和编码器的特征。
            # 对于后续层，使用的是前一层的中间参考点（inter_references[lvl - 1]），这些参考点在每个解码层中被逐步精细化。
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            #将其从 [0, 1] 范围内的值转换回原始的坐标空间
            reference = inverse_sigmoid(reference)
            #预测
            #使用从相应的解码器层 (hs[lvl]) 得到的隐藏状态，通过一个线性层 (self.class_embed[lvl]) 来预测每个查询的类别。
            outputs_class = self.class_embed[lvl](hs[lvl])
            #预测边界框的位置和大小
            tmp = self.bbox_embed[lvl](hs[lvl])
            #这几行代码处理边界框的调整。
            #如果参考点有四个维度，表明它包含了位置和尺寸信息（通常在两阶段模型中使用），则直接与预测结果相加。
            #如果只有两个维度，只调整位置信息（中心点 x, y），单阶段模型的处理方式。
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1] # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]  # [batch_size, time, num_queries_per_frame, 4]
        #pred_logits 模型为每个查询预测的类别分数（logits）。这些分数通常在多类分类问题中用于表示模型对每个类别的预测信心

        # Segmentation
        #features,memory都是list
        nf = t
        mask_features = cp.checkpoint(self.pixel_decoder, features, text_features, pos, memory, nf)
        # mask_features = self.pixel_decoder(features, text_features, pos, memory, nf=t) # [batch_size*time, c, out_h, out_w]
        mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t)


        #if self.training:t为5即5帧
        #if self.valid:t是选择整个video的帧数十几帧
        # 训练阶段，传入参数为每帧的frame和mask计算score取score最高的

        # dynamic conv
        # 动态卷积的引入是为了利用深层特征和动态调整的优势，实现对视频帧中对象的更精确分割
        # 传统的卷积网络使用固定的卷积核不足以捕捉视频中对象动态变化的复杂性。动态卷积通过为每个查询生成特定的卷积核参数，
        # 使得模型能够根据当前帧的具体内容调整其响应，从而更好地处理视频中的时间变化和对象运动
        # 每个解码层的隐藏状态 hs[lvl] 包含了当前帧特征的丰富信息，这些信息被用来生成动态卷积核
        # 参考点 lvl_references 表示每个查询的空间定位，这些参考点被用来指导分割掩码的空间位置ssa

        #
        outputs_seg_masks = []
        for lvl in range(hs.shape[0]):
            #接收来自每个解码层的隐藏状态 hs[lvl]。输出的动态参数直接用于生成卷积核，这些卷积核将应用于特征图以生成分割掩码
            dynamic_mask_head_params = self.controller(hs[lvl])   # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t)
            # outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
            # outputs_seg_mask = cp.checkpoint(self.dynamic_mask_with_coords, mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_mask = cp.checkpoint(self._compute_dynamic_mask_with_coords, mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> b t q h w', t=t)
            outputs_seg_masks.append(outputs_seg_mask)
        out['pred_masks'] = outputs_seg_masks[-1]  # [batch_size, time, num_queries_per_frame, out_h, out_w]
        print("out---------",out['pred_masks'].shape)


        out_nh,out_nw = out['pred_masks'].shape[-2:]
        del mask_features

        # generate outputs, for DVS and YTVOS
        pred_masks = rearrange(outputs_seg_masks[-1], 'b t q h w -> b q t h w', t=t)  # b q t h w
        #[0]第一个batch，只有一个batch 模型预测当前类别存在的置信度
        pred_scores = out["pred_logits"][0].sigmoid()  # t q c  

        pred_scores = pred_scores.mean(0)#[q, c]每个查询在所有时间帧上的平均置信度
        max_scores, _ = pred_scores.max(-1)#q 为了找到哪一个查询在所有类别中具有最高的平均置信度

        _, max_ind = max_scores.max(-1)

        #找到最佳查询后，代码更新所有输出，只保留这个查询的预测结果
        #通过 [:, :, max_ind:max_ind + 1] 索引，代码过滤出置信度最高的那个查询的类别预测和边界框预测。
        #这意味着模型的最终输出将只包含对单个查询的预测，即在整个批次和所有时间帧中表现最好的查询。
        out["pred_logits"] = out["pred_logits"][:, :, max_ind:max_ind + 1]  # b t q c
        out["pred_boxes"] = out["pred_boxes"][:, :, max_ind:max_ind + 1]  # b t q 4
        #选中最佳查询后变为bthw
        pred_masks = pred_masks[:, max_ind]  # b t h w
        # if pad_div != 1:
        #     tar_h, tar_w = samples.tensors.shape[-2:]
        #     pred_masks = F.interpolate(pred_masks, size=(tar_h, tar_w), mode='bilinear', align_corners=False)
        out["pred_masks"] = pred_masks.unsqueeze(2)  # b t q h w
        print("out",out["pred_masks"].shape)
        # memorize image feature and padded image for prop
        # out["features"] = [x.tensors for x in features]  # [bt c h w]
        out["images"] = rearrange(samples.tensors, '(b t) c h w -> b t c h w', t=t)   # b t c h w

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks)
        
        #如果模型不在训练模式，将存储倒数第二层的参考点以供可视化使用
        if not self.training:
            # for visualization
            inter_references = inter_references[-2, :, :, :2]  # [batch_size*time, num_queries_per_frame, 2]
            inter_references = rearrange(inter_references, '(b t) q n -> b t q n', b=b, t=t) 
            out['reference_points'] = inter_references  # the reference points of last layer input
        



        # new 
        all_pred_masks_vos = []
        clip_len = out["images"].size(1)
        origin_h, origin_w = out["images"].shape[-2:]
        pred_logits = out["pred_logits"][0]
        pred_masks = out["pred_masks"][0]
        
        # according to pred_logits, select the query index
        pred_scores = pred_logits.sigmoid() # [t, q, k]
        pred_scores_sort = pred_scores
        pred_scores = pred_scores.mean(0)   # [q, K]
        max_scores, _ = pred_scores.max(-1) # [q,]
        _, max_ind = max_scores.max(-1)     # [1,]
        max_inds = max_ind.repeat(clip_len)#max_inds 重复该索引以匹配clip长度，确保可以为剪辑中的每一帧应用相同的查询索引max_inds tensor([0, 0, 0, 0, 0], device='cuda:0')
        print("max_inds",max_inds)
        pred_masks = pred_masks[range(clip_len), max_inds, ...] # [t, h, w]#使用 max_inds 选取具有最大平均分数的查询对应的掩码。
        pred_masks = pred_masks.unsqueeze(0)
        pred_masks_vos = pred_masks
        print("pred_masks",pred_masks.shape)#pred_masks torch.Size([1, 5, 72, 128])
        #pred_masks = pred_masks[:, :, :img_h, :img_w]
        #pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
        print("pred_masks",pred_masks.shape)#pred_masks torch.Size([1, 5, 288, 512]) b t h w
        #pred_masks = pred_masks.sigmoid()[0] # [t, h, w], NOTE: here mask is score

        # mask prop
        pred_scores_sort = pred_scores_sort[:, max_ind].squeeze(-1)#t q k -> t k -> t
        score_sorted, score_ind = torch.sort(pred_scores_sort, descending=True)
        print("score_sorted",score_sorted)
        print("score_ind",score_ind)

        # frame_memory, _ = torch.sort(score_ind[:round(clip_len*0.5)])# eg frame_query: tensor([0, 2], device='cuda:0') frame_memory: tensor([1, 3, 4], device='cuda:0')
        # frame_query, _ = torch.sort(score_ind[round(clip_len*0.5):])
        frame_memory = score_ind[:3]# eg frame_query: tensor([0, 2], device='cuda:0') frame_memory: tensor([1, 3, 4], device='cuda:0')
        # frame_query, _ = score_ind[round(clip_len*0.5):]

        #refer query
        # q1 = score_ind[0]
        # q2 = score_ind[1]
        # q3 = score_ind[2]
        # print("q1:", q1)
        # print("q2:", q2)    
        # print("q3:", q3)    
        #frame_memory_refer, _ = torch.sort(score_ind[:3])# eg frame_query: tensor([0, 2], device='cuda:0') frame_memory: tensor([1, 3, 4], device='cuda:0')
        # print("frame_memory_refer",frame_memory_refer)
        pred_masks_vos_q1 = pred_masks_vos.unsqueeze(2)
        # pred_masks_vos_q1 = pred_masks_vos_q1[:,frame_memory_refer[0]]
        print("pred_masks_vos_q1",pred_masks_vos_q1.shape)
        torch.cuda.empty_cache()
        print("12",frame_memory[0],frame_memory[1])
        print("13",frame_memory[0],frame_memory[2])
        # aff12 = self.compute_affinity_refer(pred_masks_vos_q1[:,frame_memory[0]],pred_masks_vos_q1[:,frame_memory[1]])
        # aff13 = self.compute_affinity_refer(pred_masks_vos_q1[:,frame_memory[0]],pred_masks_vos_q1[:,frame_query[0]])
        # aff21 = self.compute_affinity_refer(pred_masks_vos_q1[:,frame_memory[1]],pred_masks_vos_q1[:,frame_memory[0]])
        # aff23 = self.compute_affinity_refer(pred_masks_vos_q1[:,frame_memory[1]],pred_masks_vos_q1[:,frame_query[0]])
        aff12 = cp.checkpoint(self.compute_affinity_refer,pred_masks_vos_q1[:,frame_memory[0]],pred_masks_vos_q1[:,frame_memory[1]])
        torch.cuda.empty_cache()
        aff13 = cp.checkpoint(self.compute_affinity_refer,pred_masks_vos_q1[:,frame_memory[0]],pred_masks_vos_q1[:,frame_memory[2]])
        # aff21 = cp.checkpoint(self.compute_affinity_refer,pred_masks_vos_q1[:,frame_memory[1]],pred_masks_vos_q1[:,frame_memory[0]])
        # aff23 = cp.checkpoint(self.compute_affinity_refer,pred_masks_vos_q1[:,frame_memory[1]],pred_masks_vos_q1[:,frame_query[0]])

        # 清理显存
        torch.cuda.empty_cache()
        # 查看GPU总显存
        total_memory = torch.cuda.get_device_properties(0).total_memory

        # 查看已使用显存
        allocated_memory = torch.cuda.memory_allocated(0)

        # 查看保留显存
        reserved_memory = torch.cuda.memory_reserved(0)

        # 查看剩余显存
        free_memory = total_memory - reserved_memory

        print(f"Total Memory: {total_memory / (1024 ** 2):.2f} MiB")
        print(f"Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MiB")
        print(f"Reserved Memory: {reserved_memory / (1024 ** 2):.2f} MiB")
        print(f"Free Memory: {free_memory / (1024 ** 2):.2f} MiB")
        # aff31 = self.compute_affinity_refer(pred_masks_vos_q1[:,frame_memory_refer[2]],pred_masks_vos_q1[:,frame_memory_refer[0]])
        # aff32 = self.compute_affinity_refer(pred_masks_vos_q1[:,frame_memory_refer[2]],pred_masks_vos_q1[:,frame_memory_refer[1]])
        # score1 = cp.checkpoint(self.compute_score_refer,aff12,aff13)
        # score2 = cp.checkpoint(self.compute_score_refer,aff21,aff23)
        score1 = self.compute_score_refer(aff12,aff13)# q1
        # score2 = self.compute_score_refer(aff21,aff23)# q2
        # score3 = self.compute_score_refer(aff31,aff32)# q3


        if score1>0.8:
            print("score1",score1)   
            # q12 = cp.checkpoint(self.compute_bmm_refer,pred_masks_vos_q1[:,frame_memory[1]],aff12)   
            q12 = self.compute_bmm_refer(pred_masks_vos_q1[:,frame_memory[1]],aff12)# b cv h w
            q12 = torch.cat([q12,pred_masks_vos_q1[:,frame_memory[0]]],dim = 1)# b cv+cv/2 h w
            # q12 = cp.checkpoint(self.conv_refer,q12)
            q12 = self.conv_refer(q12)# b cv h w
            aff123 = self.compute_affinity_refer(q12,pred_masks_vos_q1[:,frame_memory[2]])
            q123 = self.compute_bmm_refer(pred_masks_vos_q1[:,frame_memory[2]],aff123)
            q123 = torch.cat([q123,pred_masks_vos_q1[:,frame_memory[0]]],dim = 1)
            q123 = self.conv_refer(q123)
            # q13 = cp.checkpoint(self.compute_bmm_refer,pred_masks_vos_q1[:,frame_query[2]],aff13)   
            # q13 = self.compute_bmm_refer(pred_masks_vos_q1[:,frame_query[2]],aff13)            
            # q13 = torch.cat([q13,pred_masks_vos_q1[:,frame_memory[0]]],dim = 1)
            # # q13 = cp.checkpoint(self.conv_refer,q13)
            # q13 = self.conv_refer(q13)
            # q123 = (q12+q13)/2
            # print("q123",q123.shape)
            # print("pred_masks_vos",pred_masks_vos.shape)
            new_pred_masks_vos = pred_masks_vos.clone()
            new_pred_masks_vos[:, frame_memory[0], :, :] = q123.squeeze(1)
            pred_masks_vos = new_pred_masks_vos
            # pred_masks_vos[:, frame_memory[0], :, :] = q123.squeeze(1)
            # print("pred_masks_vos",pred_masks_vos.shape)
            # print("Replaced slice equals B:", torch.all(pred_masks_vos[:, frame_memory_refer[0], :, :] == q123.squeeze(1)).item())

        # if score2>0.8:
        #     print("score2",score2)
        #     # q21 = cp.checkpoint(self.compute_bmm_refer,pred_masks_vos_q1[:,frame_memory[0]],aff21)   
        #     q21 = self.compute_bmm_refer(pred_masks_vos_q1[:,frame_memory[0]],aff21)
        #     q21 = torch.cat([q21,pred_masks_vos_q1[:,frame_memory[1]]],dim = 1)# b cv+cv/2 h w
        #     # q21 = cp.checkpoint(self.conv_refer,q21)
        #     q21 = self.conv_refer(q21)

        #     # q23 = cp.checkpoint(self.compute_bmm_refer,pred_masks_vos_q1[:,frame_query[2]],aff23)
        #     q23 = self.compute_bmm_refer(pred_masks_vos_q1[:,frame_query[2]],aff23)
        #     q23 = torch.cat([q23,pred_masks_vos_q1[:,frame_memory[1]]],dim = 1)
        #     # q23 = cp.checkpoint(self.conv_refer,q23)
        #     q23 = self.conv_refer(q23)
        #     q213 = (q21+q23)/2
        #     new_pred_masks_vos = pred_masks_vos.clone()
        #     new_pred_masks_vos[:, frame_memory[1], :, :] = q213.squeeze(1)
        #     pred_masks_vos = new_pred_masks_vos
            # pred_masks_vos[:, frame_memory[1], :, :] = q213.squeeze(1)
        # if score3>0.8:
        #     print("score3",score3)
        #     q31 = self.compute_bmm_refer(pred_masks_vos_q1[:,frame_memory_refer[0]],aff31)
        #     q31 = torch.cat([q31,pred_masks_vos_q1[:,frame_memory_refer[2]]],dim = 1)
        #     q31 = self.conv_refer(q31)
        #     q32 = self.compute_bmm_refer(pred_masks_vos_q1[:,frame_memory_refer[1]],aff32)
        #     q32 = torch.cat([q32,pred_masks_vos_q1[:,frame_memory_refer[2]]],dim = 1)
        #     q32 = self.conv_refer(q32)
        #     q312 = (q31+q32)/2
        #     pred_masks_vos[:, frame_memory_refer[2], :, :] = q312.squeeze(1)

            
        pred_masks_vos= pred_masks_vos.unsqueeze(2)
        # print("pred_masks_vos222222222222222222222222221",pred_masks_vos.shape)
        out["pred_masks"] = pred_masks_vos
        
        
        return out
    
  
    def _compute_transformer(self, srcs, text_embed, masks, poses, query_embeds):
        return self.transformer(srcs, text_embed, masks, poses, query_embeds)
    def _compute_dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        return self.dynamic_mask_with_coords(mask_features, mask_head_params, reference_points, targets)
    # def forward_eval(mem_feat, mem_mask, mem_image, query_feat, query_image, frame_memory, frame_query, topk=20):
    #         return self.voshead.forward_eval(mem_feat, mem_mask, mem_image, query_feat, query_image, frame_memory, frame_query, topk)
    # def forward_eval_wrapped(*args):
    #     mem_feat, mem_mask, mem_image, query_feat, query_image, frame_memory, frame_query = args
    #     return self.forward_eval(mem_feat, mem_mask, mem_image, query_feat, query_image, frame_memory, frame_query, topk=20)



        

    def compute_score_refer(self, aff1, aff2):
        intersection = torch.logical_and(aff1, aff2)
        union = torch.logical_or(aff1, aff2)
        jaccard_index = intersection.sum().float() / union.sum()
 
        print("Jaccard 系数:", jaccard_index.item())  # 使用.item()从单个值的张量中提取标量
        return jaccard_index
    def compute_bmm_refer(self, m, aff):
        B, CK, H, W = m.shape
        m = m.view(B, CK, H*W)
        mout_refer = torch.bmm(m, aff)
        mout_refer = mout_refer.view(B, CK, H, W)
        return mout_refer

    def compute_affinity_refer(self, qk, mk):
        B, CK, H, W = mk.shape
      
        mi = mk.view(B, CK, H*W) #将时间和空间维度合并。
        mi = torch.transpose(mi, 1, 2) # B * THW * CK
        #
        #除以 math.sqrt(CK) 进行归一化，有助于稳定学习过程
        qi = qk.view(B, CK, H*W) / math.sqrt(CK)  # B * CK * HW

        # 计算输入特征与查询特征之间的亲和度,相似度
        affinity = torch.bmm(mi, qi)  # B, THW, HW
        affinity = F.softmax(affinity, dim=1)  # B, THW, HW

  
        return affinity

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c} 
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            #多个样本
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length]

            text_features = encoded_text.last_hidden_state 
            text_features = self.resizer(text_features)    
            text_masks = text_attention_mask              
            text_features = NestedTensor(text_features, text_masks) # NestedTensor

            text_sentence_features = encoded_text.pooler_output  
            text_sentence_features = self.resizer(text_sentence_features)  
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features, text_sentence_features

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        # # 计算所有帧中的查询总数。
        _, num_queries = reference_points.shape[:2]  
        q = num_queries // t  #每帧的查询数

        # 准备图像尺寸的参考点（模型的输入尺寸）
        new_reference_points = [] 
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0) 
            tmp_reference_points = reference_points[i] * scale_f[None, :] 
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0) 
        # [batch_size, time * num_queries_per_frame, 2], in image size
        reference_points = new_reference_points  

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q) 
            # 计算每个位置的坐标
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride)
            # 计算相对坐标 
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                                    locations.reshape(1, 1, 1, h, w, 2) # [batch_size, time, num_queries_per_frame, h, w, 2]
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3, 4) # [batch_size, time, num_queries_per_frame, 2, h, w]

            # 将相对坐标合并到特征中
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            # 如果未启用相对坐标功能，直接扩展特征维度
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
        
        # 重塑mask特征，准备卷积
        mask_features = mask_features.reshape(1, -1, h, w) 

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1) 
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic mask conv
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0]) 
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        # upsample predicted masks
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  # [batch_size, time * num_queries_per_frame, h, w]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        #根据每个目标的特定情况动态调整卷积核，以生成精确的目标分割掩膜。
        #通过分组卷积实现动态调整，每组处理一组特定的查询，确保每个目标的处理是独立和特定的。
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''

        # features：输入特征图，维度为 [batch_size, channels, height, width]。
        # weights：动态生成的卷积核权重列表，每一项对应一个卷积层的权重。
        # biases：与权重对应的偏置列表。
        # num_insts：需要进行卷积的实例数量，通常等于批次大小乘以时间帧数乘以每帧的查询数。

        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
                #groups=num_insts：使用分组卷积，每个实例的特征独立进行卷积。这是动态卷积的关键，允许每个查询有其专属的卷积核。
            )
            #如果不是最后一层，使用 F.relu 作为激活函数，增加非线性，帮助捕捉复杂的特征。
            if i < n_layers - 1:
                x = F.relu(x)
        return x


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4 
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65 
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else: 
            num_classes = 91 # for coco
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args) 
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = ReferFormer(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        mask_dim=args.mask_dim,
        dim_feedforward=args.dim_feedforward,
        controller_layers=args.controller_layers,
        dynamic_mask_channels=args.dynamic_mask_channels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        rel_coord=args.rel_coord
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks: # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
            num_classes, 
            matcher=matcher,
            weight_dict=weight_dict, 
            eos_coef=args.eos_coef, 
            losses=losses,
            focal_alpha=args.focal_alpha)
    criterion.to(device)

    # postprocessors, this is used for coco pretrain but not for rvos
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors


