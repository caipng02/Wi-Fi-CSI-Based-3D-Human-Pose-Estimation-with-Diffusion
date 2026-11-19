"""DT-Pose encoder and the two Phase 1 decoders, using only PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F


JOINTS = 17
WIDTH = 256
HEADS = 4
PATCHES = 285
CSI_SHAPE = (3, 114, 10)
EDGES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
         (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
         (8, 14), (14, 15), (15, 16))


class ViTAttention(nn.Module):
    """State layout of the timm attention used by the DT-Pose encoder."""

    def __init__(self, bias=False):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=bias)
        self.proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, HEADS, width // HEADS).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, width))


class FeedForward(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, hidden)
        self.fc2 = nn.Linear(hidden, WIDTH)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class ViTBlock(nn.Module):
    def __init__(self, hidden, bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH)
        self.attn = ViTAttention(bias)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.mlp = FeedForward(hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DTEncoder(nn.Module):
    """Unmasked token path from DT-Pose model.py, before decoder mean pooling."""

    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, WIDTH))
        self.pos_embedding = nn.Parameter(torch.empty(PATCHES, 1, WIDTH))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.patchify = nn.Conv2d(3, WIDTH, kernel_size=2, stride=2)
        self.transformer = nn.Sequential(*(ViTBlock(4 * WIDTH) for _ in range(4)))
        self.layer_norm = nn.LayerNorm(WIDTH)

    def patches(self, csi):
        if csi.ndim != 4 or csi.shape[1:] != CSI_SHAPE:
            raise ValueError(f"Expected CSI [B, 3, 114, 10], got {tuple(csi.shape)}")
        patches = self.patchify(csi).flatten(2).transpose(1, 2)
        return patches + self.pos_embedding.transpose(0, 1)

    def encode(self, patches):
        cls = self.cls_token.expand(patches.size(0), -1, -1)
        return self.layer_norm(self.transformer(torch.cat((cls, patches), dim=1)))

    def forward(self, csi):
        return self.encode(self.patches(csi))[:, 1:]


class GraphConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(WIDTH, WIDTH)

    def forward(self, x, adjacency):
        return F.relu(self.fc(adjacency @ x))


class DTPoseDecoderBlock(nn.Module):
    """DT-Pose's post-normalization joint block."""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.linear1 = nn.Linear(WIDTH, WIDTH)
        self.linear2 = nn.Linear(WIDTH, WIDTH)
        self.norm1 = nn.LayerNorm(WIDTH)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        attention, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout1(attention))
        return self.norm2(x + self.dropout2(self.linear2(F.relu(self.linear1(x)))))


class DTPoseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_prompt = nn.Parameter(torch.empty(WIDTH, JOINTS))
        nn.init.trunc_normal_(self.pose_prompt, std=0.02)
        adjacency = torch.zeros(JOINTS, JOINTS)
        for a, b in EDGES:
            adjacency[a, b] = adjacency[b, a] = 1
        self.register_buffer("adjacency", adjacency, persistent=False)
        self.graph_layers = nn.ModuleList(GraphConvLayer() for _ in range(3))
        self.joint_layers = nn.Sequential(*(DTPoseDecoderBlock() for _ in range(3)))
        self.pose_head = nn.Sequential(nn.Linear(WIDTH, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, patches):
        joints = patches.mean(1, keepdim=True) + self.pose_prompt.T.unsqueeze(0)
        for layer in self.graph_layers:
            joints = layer(joints, self.adjacency)
        joints = self.joint_layers(joints)
        return self.pose_head(joints), joints


class GRCPoseDecoder(nn.Module):
    """mmDiff-style joint-token attention with CSI patch tokens."""

    def __init__(self):
        super().__init__()
        self.joint_tokens = nn.Parameter(torch.empty(1, JOINTS, WIDTH))
        nn.init.trunc_normal_(self.joint_tokens, std=0.02)
        self.transformer = nn.Sequential(ViTBlock(512, bias=True), ViTBlock(512, bias=True))
        self.norm = nn.LayerNorm(WIDTH)
        self.projection = nn.Linear(WIDTH, 64)
        self.pose_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, patches):
        tokens = torch.cat((self.joint_tokens.expand(patches.size(0), -1, -1), patches), dim=1)
        joints = self.norm(self.transformer(tokens)[:, :JOINTS])
        features = self.projection(joints)
        return self.pose_head(features), features


class Phase1(nn.Module):
    def __init__(self, variant, pelvis_index):
        super().__init__()
        self.encoder = DTEncoder()
        self.decoder = {"baseline": DTPoseDecoder, "grc": GRCPoseDecoder}[variant]()
        self.pelvis_index = pelvis_index

    def forward(self, csi):
        pose, features = self.decoder(self.encoder(csi))
        root = pose[:, self.pelvis_index:self.pelvis_index + 1]
        return pose - root, features
