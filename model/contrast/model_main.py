from timm.models import create_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.contrast.slots import ScouterAttention, vis
from model.contrast.position_encode import build_position_encoding


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(args.base_model, pretrained=True,
                        num_classes=args.num_classes)

    bone.global_pool = Identical()
    bone.fc = Identical()
    # fix_parameter(bone, [""], mode="fix")
    # fix_parameter(bone, ["layer4", "layer3"], mode="open")
    return bone


class MainModel(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        hidden_dim = 128
        num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.back_bone = load_backbone(args)
        self.activation = nn.Tanh()
        self.vis = vis

        if not self.pre_train:
            self.conv1x1 = nn.Conv2d(self.num_features, hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            self.norm = nn.BatchNorm2d(hidden_dim)
            self.position_emb = build_position_encoding('sine', hidden_dim=hidden_dim)
            self.slots = ScouterAttention(args, hidden_dim, num_concepts, vis=self.vis)
            self.scale = 1
            self.cls = torch.nn.Linear(num_concepts, num_classes)
        else:
            self.fc = nn.Linear(self.num_features, num_classes)
            self.drop_rate = 0

    # 获取最后全连接层的权重
    def get_last_layer_weights(self):
        return self.cls.weight.data

    def forward(self, x, weight=None, things=None, return_cpt=False, loc="vis", loc2="vis_pp", mask_size=224):
        # Input x shape: torch.Size([256, 3, 224, 224])
        x = self.back_bone(x)
        # After back_bone, x shape: torch.Size([256, 512, 7, 7])
        features = x
        # x = x.view(x.size(0), self.num_features, self.feature_size, self.feature_size)

        if not self.pre_train:
            x = self.conv1x1(x)
            # After conv1x1, x shape: torch.Size([256, 128, 7, 7])
            x = self.norm(x)
            # After norm, x shape: torch.Size([256, 128, 7, 7])
            x = torch.relu(x)
            pe = self.position_emb(x)
            # After position_emb, pe shape: torch.Size([256, 128, 7, 7])
            x_pe = x + pe
            # After adding position encoding, x_pe shape: torch.Size([256, 128, 7, 7])

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0, 2, 1))
            # After reshaping and permuting x, x shape: torch.Size([256, 49, 128])
            x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
            # After reshaping and permuting x_pe, x_pe shape: torch.Size([256, 49, 128])
            updates, attn = self.slots(x_pe, x, loc, loc2, mask_size, weight, things)
            # After ScouterAttention, updates shape: torch.Size([256, 20, 128]) , attn shape: torch.Size([256, 20, 49])
            if self.args.cpt_activation == "att":
                cpt_activation = attn
            else:
                cpt_activation = updates
            attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)
            # After computing attn_cls, attn_cls shape: torch.Size([256, 20])
            cpt = self.activation(attn_cls)
            # After applying activation to attn_cls, cpt shape: torch.Size([256, 20])
            cls = self.cls(cpt)
            # After final linear layer, cls shape: torch.Size([256, 50])
            if return_cpt:
                return (cpt - 0.5) * 2, cls, attn, updates, self.get_last_layer_weights()
            else:
                return (cpt - 0.5) * 2, cls, attn, updates
        else:
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
            # After adaptive max pool and squeeze, x shape: torch.Size([256, 512])
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.fc(x)
            # After final linear layer in pre_train mode, x shape: torch.Size([256, 50])
            return x, features


class MainModel2(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel2, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.back_bone = load_backbone(args)

    def forward(self, x):
        x = self.back_bone(x)
        features = x

        return features


# if __name__ == '__main__':
#     model = MainModel()
#     inp = torch.rand((2, 1, 224, 224))
#     pred, out, att_loss = model(inp)
#     print(pred.shape)
#     print(out.shape)


