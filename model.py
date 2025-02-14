from transformers import LlamaForCausalLM, LlamaTokenizer
import torch.nn as nn
from torchvision.models import convnext_large
import torch
import os

# 设置Hugging Face Hub的访问令牌，用于下载模型
os.environ["HUGGINGFACE_HUB_TOKEN"] ='你自己的令牌'
# 加载DINOv2的ViT-S14模型
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

#dinov2_vits14=torch.load('./dinov2/dinov2_vits14.pth')
#convnext=torch.load('./convnext/convnext_large.pth')

# ViT特征提取器类
class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vits14'):
        super(ViTFeatureExtractor, self).__init__()
        # 加载DINOv2的ViT-S14模型
        self.vit = dinov2_vits14

    def forward(self, x):
        features = self.vit.get_intermediate_layers(x, n=12)
        # 每层都是 [batch_size, seq_len, hidden_dim]
        # 对 seq_len 做均值池化[batch_size, hidden_dim]
        layer_pool = [layer.mean(dim=1) for layer in features]
        vit_features = torch.cat(layer_pool, dim=-1)  # [batch_size, 12*hidden_dim]
        return vit_features

# CNN特征提取器类
class CNNFeatureExtractor(nn.Module):
    def __init__(self, model_name='convnext_large'):
        super(CNNFeatureExtractor, self).__init__()
        # 加载预训练的ConvNeXt Large模型
        self.clip_model = convnext_large(pretrained=True)
        # 冻结CNN主干的参数，使其在训练过程中不更新
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 直接获取模型的输出特征
        cnn_features = self.clip_model(x)
        return cnn_features

# 视觉投影层类，用于将ViT和CNN的特征投影到同一维度
class VisualProjection(nn.Module):
    def __init__(self, vit_dim, cnn_dim, output_dim):
        super(VisualProjection, self).__init__()
        # 定义线性投影层
        self.proj = nn.Linear(vit_dim + cnn_dim, output_dim)

    def forward(self, vit_feat, cnn_feat):
        # 拼接ViT和CNN的特征
        combined_feat = torch.cat([vit_feat, cnn_feat], dim=1)
        # 进行投影操作
        projected_feat = self.proj(combined_feat)
        return projected_feat

# EarthGPT模型类，实现视觉语言融合和初步的训练功能
class EarthGPT(nn.Module):
    def __init__(self, visual_proj_dim, llm_name='meta-llama/Llama-2-7b-chat-hf'):
        super(EarthGPT, self).__init__()
        # 加载Llama-2的分词器
        self.tokenizer = LlamaTokenizer.from_pretrained(llm_name)
        # 加载Llama-2的因果语言模型
        self.llm = LlamaForCausalLM.from_pretrained(llm_name)
        # 解冻self-attention和RMSNorm层的参数，使其在训练过程中可更新
        for name, param in self.llm.named_parameters():
            if'self_attn' in name or 'rms_norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # 定义视觉特征到语言模型隐藏层大小的投影层
        self.vis_proj = nn.Linear(visual_proj_dim, self.llm.config.hidden_size)

    def forward(self, visual_feat, lang_instr):
        #1视觉特征Llama hidden_size
        vis_emb = self.vis_proj(visual_feat)#[batch_size, hidden_size]

        #2扩展维度，作为“额外的 1 个 token
        vis_emb = vis_emb.unsqueeze(1)#[batch_size, 1, hidden_size]


        #3文本分词并 embeddings
        #Llama 默认没有 pad_token，需要手动指定或只输入单条文本
        lang_tokens = self.tokenizer(lang_instr, return_tensors='pt')
        text_emb = self.llm.model.embed_tokens(lang_tokens.input_ids)#[batch_size, seq_len, hidden_size]


        #4在序列维度拼接[batch_size, 1 + seq_len, hidden_size]
        input_emb = torch.cat([vis_emb, text_emb], dim=1)

        #5计算 loss，需要与 labels 对齐：
        ignore_label = torch.full(
            (lang_tokens.input_ids.size(0), 1),
            -100,
            dtype=torch.long
        )
        extended_labels = torch.cat([ignore_label, lang_tokens.input_ids], dim=1)

        #6前向传播
        outputs = self.llm(
            inputs_embeds=input_emb,
            labels=extended_labels
        )
        return outputs.loss

# 带有偏差调整的线性层类
class BiasTuningLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BiasTuningLinear, self).__init__()
        # 定义普通线性层
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # 定义可学习的偏差参数
        self.beta = nn.Parameter(torch.zeros(out_features))
        # 定义可学习的缩放参数
        self.alpha = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # 先经过普通线性层
        y = self.linear(x)
        # 应用偏差和缩放调整
        y = self.alpha * (y + self.beta)
        return y

# 统一指令调整的EarthGPT模型类，继承自EarthGPT
class EarthGPTUnified(EarthGPT):
    def __init__(self, visual_proj_dim, llm_name='meta-llama/Llama-2-7b-chat-hf'):
        super(EarthGPTUnified, self).__init__(visual_proj_dim, llm_name)
        # 替换语言模型中的线性层为带有偏差调整的线性层
        self.llm = self.replace_linear_layers(self.llm)

    def replace_linear_layers(self, module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, nn.Linear):
                # 将普通线性层替换为带有偏差调整的线性层
                setattr(module, name, BiasTuningLinear(sub_module.in_features, sub_module.out_features, sub_module.bias is not None))
            else:
                # 递归处理子模块
                self.replace_linear_layers(sub_module)
        return module


