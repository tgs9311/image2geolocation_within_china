import torch
import torch.nn.functional as F

# haversine距离函数，返回 km
def haversine_distance(pred_coords, true_coords):
    # pred_coords 和 true_coords 均为 (B, 2)
    R = 6371  # 地球半径，单位：km
    pred_rad = torch.deg2rad(pred_coords)
    true_rad = torch.deg2rad(true_coords)
    delta = pred_rad - true_rad
    dlat = delta[:, 0]
    dlon = delta[:, 1]
    a = torch.sin(dlat / 2) ** 2 + torch.cos(pred_rad[:, 0]) * torch.cos(true_rad[:, 0]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(a))
    return R * c  # 返回的是 km


def HaversineCoordLoss(logits, true_coords, center_tensor):
    """
    logits: (B, 37) 模型未归一化输出
    true_coords: (B, 2) 每个样本的真实坐标（纬度, 经度），单位是度
    center_tensor: (37, 2) 每个省的中心坐标
    返回:
    - loss: 标量
    """
    probs = F.softmax(logits, dim=-1)  # (B, 37)
    pred_coords = probs @ center_tensor  # (B, 2)，每行是 weighted sum of centers

    distances = haversine_distance(pred_coords, true_coords)  # (B,)
    return distances.mean()