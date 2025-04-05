# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.structures.bbox import HorizontalBoxes

from mmyolo.registry import MODELS
class WIoU_Scale:
    ''' monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean'''

    iou_mean = 1.
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1

def bbox_overlaps(pred: torch.Tensor,
                  target: torch.Tensor,
                  iou_mode: str = 'ciou',
                  bbox_format: str = 'xywh',
                  siou_theta: float = 4.0,
                  gamma: float = 0.5,
                  eps: float = 1e-7,
                  focal: bool = False,) -> torch.Tensor:
    r"""Calculate overlap between two set of bboxes.
    `Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
    difference in the way the alpha parameter is computed.

    mmdet version:
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    YOLOv5 version:
        alpha = v / (v - ious + (1 + eps)

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        siou_theta (float): siou_theta for SIoU when calculate shape cost.
            Defaults to 4.0.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: shape (n, ).
    """
    assert iou_mode in ('iou', 'ciou', 'giou', 'siou','diou','eiou','innersiou','innerciou','shapeiou','wiou')
    assert bbox_format in ('xyxy', 'xywh')
    if bbox_format == 'xywh':
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)

    bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
    bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
    bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
    bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]

    # Overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) -
               torch.max(bbox1_x1, bbox2_x1)).clamp(0) * \
              (torch.min(bbox1_y2, bbox2_y2) -
               torch.max(bbox1_y1, bbox2_y1)).clamp(0)

    # Union
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = (w1 * h1) + (w2 * h2) - overlap + eps

    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
    enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    enclose_w = enclose_wh[..., 0]  # cw
    enclose_h = enclose_wh[..., 1]  # ch

    if iou_mode == 'ciou':
        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                           (bbox1_y1 + bbox1_y2))**2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)

        # Width and height ratio (v)
        wh_ratio = (4 / (math.pi**2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

        if focal:
            # focal-CIOU
            ious = torch.pow(ious, gamma)*(ious - ((rho2 / enclose_area) + (alpha * wh_ratio)))
            # cious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

            # return cious, ious, gamma
        else:
            #CIOU
            ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

    elif iou_mode == 'giou':
        # GIoU = IoU - ( (A_c - union) / A_c )
        convex_area = enclose_w * enclose_h + eps  # convex area (A_c)
        ious = ious - (convex_area - union) / convex_area

    elif iou_mode == 'siou':
        # SIoU: https://arxiv.org/pdf/2205.12740.pdf
        # SIoU = IoU - ( (Distance Cost + Shape Cost) / 2 )

        # calculate sigma (σ):
        # euclidean distance between bbox2(pred) and bbox1(gt) center point,
        # sigma_cw = b_cx_gt - b_cx
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        # sigma_ch = b_cy_gt - b_cy
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        # sigma = √( (sigma_cw ** 2) - (sigma_ch ** 2) )
        sigma = torch.pow(sigma_cw**2 + sigma_ch**2, 0.5)

        # choose minimize alpha, sin(alpha)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha,
                                sin_beta)

        # Angle cost = 1 - 2 * ( sin^2 ( arcsin(x) - (pi / 4) ) )
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # Distance cost = Σ_(t=x,y) (1 - e ^ (- γ ρ_t))
        rho_x = (sigma_cw / enclose_w)**2  # ρ_x
        rho_y = (sigma_ch / enclose_h)**2  # ρ_y
        gamma = 2 - angle_cost  # γ
        distance_cost = (1 - torch.exp(-1 * gamma * rho_x)) + (
            1 - torch.exp(-1 * gamma * rho_y))

        # Shape cost = Ω = Σ_(t=w,h) ( ( 1 - ( e ^ (-ω_t) ) ) ^ θ )
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # ω_w
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # ω_h
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w),
                               siou_theta) + torch.pow(
                                   1 - torch.exp(-1 * omiga_h), siou_theta)

        ious = ious - ((distance_cost + shape_cost) * 0.5)
    elif iou_mode == 'diou':
        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                           (bbox1_y1 + bbox1_y2))**2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)
        ious = ious - ((rho2 / enclose_area))
    elif iou_mode == "eiou":

        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                           (bbox1_y1 + bbox1_y2))**2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)
        rho_w2 = ((bbox2_x2 - bbox2_x1) - (bbox1_x2 - bbox1_x1)) ** 2
        rho_h2 = ((bbox2_y2 - bbox2_y1) - (bbox1_y2 - bbox1_y1)) ** 2
        cw2 = enclose_w ** 2 + eps
        ch2 = enclose_h ** 2 + eps
        ious = ious - (rho2 / enclose_area + rho_w2 / cw2 + rho_h2 / ch2)
    elif iou_mode == "innersiou":
        ratio=1.0
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        x1 = bbox1_x1 + w1_
        y1 = bbox1_y1 + h1_
        x2 = bbox2_x1 + w2_
        y2 = bbox2_y1 + h2_

        inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_ * ratio, x1 + w1_ * ratio, \
                                                             y1 - h1_ * ratio, y1 + h1_ * ratio
        inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_ * ratio, x2 + w2_ * ratio, \
                                                             y2 - h2_ * ratio, y2 + h2_ * ratio
        inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                      (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
        inner_union = w1 * ratio * h1 * ratio + w2 * ratio * h2 * ratio - inner_inter + eps
        inner_iou = inner_inter / inner_union

        cw = torch.max(bbox1_x2, bbox2_x2) - torch.min(bbox1_x1, bbox2_x1)  # convex width
        ch = torch.max(bbox1_y2, bbox2_y2) - torch.min(bbox1_y1, bbox2_y1)  # convex height
        s_cw = (bbox2_x1 + bbox2_x2 - bbox1_x1 - bbox1_x2) * 0.5 + eps
        s_ch = (bbox2_y1 + bbox2_y2 - bbox1_y1 - bbox1_y2) * 0.5 + eps
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        ious = inner_iou - 0.5 * (distance_cost + shape_cost)
    elif iou_mode == "innerciou":
        ratio=1.0
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        x1 = bbox1_x1 + w1_
        y1 = bbox1_y1 + h1_
        x2 = bbox2_x1 + w2_
        y2 = bbox2_y1 + h2_

        inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_ * ratio, x1 + w1_ * ratio, \
                                                             y1 - h1_ * ratio, y1 + h1_ * ratio
        inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_ * ratio, x2 + w2_ * ratio, \
                                                             y2 - h2_ * ratio, y2 + h2_ * ratio
        inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                      (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
        inner_union = w1 * ratio * h1 * ratio + w2 * ratio * h2 * ratio - inner_inter + eps
        inner_iou = inner_inter / inner_union

        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                           (bbox1_y1 + bbox1_y2))**2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)

        # Width and height ratio (v)
        wh_ratio = (4 / (math.pi**2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

        # innerCIoU
        ious = inner_iou - ((rho2 / enclose_area) + (alpha * wh_ratio))
    elif iou_mode == "shapeiou":
        scale = 0
        # Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
        ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        cw = torch.max(bbox1_x2, bbox2_x2) - torch.min(bbox1_x1, bbox2_x1)  # convex width
        ch = torch.max(bbox1_y2, bbox2_y2) - torch.min(bbox1_y1, bbox2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        center_distance_x = ((bbox2_x1 + bbox2_x2 - bbox1_x1 - bbox1_x2) ** 2) / 4
        center_distance_y = ((bbox2_y1 + bbox2_y2 - bbox1_y1 - bbox1_y2) ** 2) / 4
        center_distance = hh * center_distance_x + ww * center_distance_y
        distance = center_distance / c2

        # Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
        omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

        # Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU



        ious = ious - distance - 0.5 * (shape_cost)
    elif iou_mode == "wiou":
        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )
            enclose_area = enclose_w**2 + enclose_h**2 + eps

            # calculate ρ^2(b_pred,b_gt):
            # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
            # center point, because bbox format is xyxy -> left-top xy and
            # right-bottom xy, so need to / 4 to get center point.
            rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
            rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                               (bbox1_y1 + bbox1_y2))**2 / 4
            rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)
            obj = WIoU_Scale(ious)
            wise_iou_loss1 = getattr(obj,'_scaled_loss')(obj)
            wise_iou_loss2 = (1-ious)* torch.exp((rho2 / enclose_area))

            return wise_iou_loss1,wise_iou_loss2,ious.clamp(min=-1.0, max=1.0)

    return ious.clamp(min=-1.0, max=1.0)


@MODELS.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    Args:
        iou_mode (str): Options are "ciou".
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        return_iou (bool): If True, return loss and iou.
    """

    def __init__(self,
                 iou_mode: str = 'ciou',
                 bbox_format: str = 'xywh',
                 eps: float = 1e-7,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 focal: bool = False,
                 return_iou: bool = True):
        super().__init__()
        assert bbox_format in ('xywh', 'xyxy')
        assert iou_mode in ('ciou', 'siou', 'giou','diou','eiou','innersiou','innerciou','shapeiou','wiou')
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou
        self.focal = focal
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[Union[str, bool]] = None
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
                or (x, y, w, h),shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            loss or tuple(loss, iou):
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        iou = bbox_overlaps(
            pred,
            target,
            iou_mode=self.iou_mode,
            bbox_format=self.bbox_format,
            eps=self.eps)
        if type(iou) is tuple:
            loss = self.loss_weight * weight_reduce_loss(1.0 - iou[2], weight,
                                                         reduction, avg_factor)
            loss += weight_reduce_loss((iou[0]*iou[1]).mean(),weight,
                                       reduction,avg_factor)
            iou = iou[2]
        else:
            loss = self.loss_weight * weight_reduce_loss(1.0 - iou, weight,
                                                     reduction, avg_factor)

        if self.return_iou:
            return loss, iou
        else:
            return loss