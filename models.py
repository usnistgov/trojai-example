from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import copy
import numpy as np
import torch
import torch.nn
import torchvision.transforms
import torchvision.models.detection.transform
import torchvision.ops.boxes as box_ops

import transformers


def x1y1x2y2_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = torchvision.models.resnet50(pretrained=pretrained_backbone, progress=progress, norm_layer=torchvision.ops.misc.FrozenBatchNorm2d)
    backbone = torchvision.models.detection.backbone_utils._resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.detection.faster_rcnn.model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
        model.load_state_dict(state_dict)
        torchvision.models.detection._utils.overwrite_eps(model, 0.0)
    return model


class FasterRCNN(torchvision.models.detection.generalized_rcnn.GeneralizedRCNN):

    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (torchvision.models.detection.anchor_utils.AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (torchvision.ops.poolers.MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = torchvision.models.detection.rpn.RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # need to completely re-implement this init, to replace the RPN head
        #rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
        rpn=RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = torchvision.ops.poolers.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(representation_size, num_classes)

        # need to completely re-implement this init, to replace the RoIHeads
        # roi_heads = torchvision.models.detection.roi_heads.RoIHeads(
        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super().__init__(backbone, rpn, roi_heads, transform)

    def prepare_inputs(self, images, targets):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if targets is not None:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        if self.training:
            assert targets is not None

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        return images, targets, original_image_sizes

    def basic_forward(self, images, targets):
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images, targets, original_image_sizes = self.prepare_inputs(images, targets)

        losses, detections = self.basic_forward(images, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        if targets is None:
            return detections
        else:
            return losses, detections


class RoIHeads(torchvision.models.detection.roi_heads.RoIHeads):
    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, "target keypoints must of float type"

        if targets is not None:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if targets is not None:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = torchvision.models.detection.roi_heads.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if targets is not None:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if targets is not None:
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = torchvision.models.detection.roi_heads.maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}

            labels = [r["labels"] for r in result]
            masks_probs = torchvision.models.detection.roi_heads.maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if targets is not None:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if targets is not None:
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = torchvision.models.detection.roi_heads.keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}

            assert keypoint_logits is not None
            assert keypoint_proposals is not None

            keypoints_probs, kp_scores = torchvision.models.detection.roi_heads.keypointrcnn_inference(keypoint_logits, keypoint_proposals)
            for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                r["keypoints"] = keypoint_prob
                r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses


class RegionProposalNetwork(torchvision.models.detection.rpn.RegionProposalNetwork):
    def forward(
        self,
        images: torchvision.models.detection.image_list.ImageList,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = torchvision.models.detection.rpn.concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if targets is not None:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses






def ssd300_vgg16(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 91,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
):

    trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 4
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    # Use custom backbones more appropriate for SSD
    backbone = torchvision.models.vgg.vgg16(pretrained=False, progress=progress)
    if pretrained_backbone:
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.detection.backbone_urls["vgg16_features"], progress=progress)
        backbone.load_state_dict(state_dict)

    backbone = torchvision.models.detection.ssd._vgg_extractor(backbone, False, trainable_backbone_layers)
    anchor_generator = torchvision.models.detection.anchor_utils.DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300],
    )

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    if pretrained:
        weights_name = "ssd300_vgg16_coco"
        if torchvision.models.detection.ssd.model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = torchvision.models.detection.ssd.load_state_dict_from_url(torchvision.models.detection.ssd.model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    return model


class SSD(torchvision.models.detection.SSD):

    def prepare_inputs(self, images, targets):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if targets is not None:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        if self.training:
            assert targets is not None

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        return images, targets, original_image_sizes

    def basic_forward(self, images, targets):
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        loss_dict = dict()
        if targets is not None:
            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                    )
                    continue

                match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

            loss_dict = self.compute_loss(targets, head_outputs, anchors, matched_idxs)

        detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)

        return loss_dict, detections

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images, targets, original_image_sizes = self.prepare_inputs(images, targets)

        losses, detections = self.basic_forward(images, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if targets is None:
            return detections
        else:
            return losses, detections



def detr_model():
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    return model


class DetrForObjectDetection(transformers.DetrForObjectDetection):
    # based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        # self.model = transformers.DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        self.transform = transformers.DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        # If you have more than 1 image per batch (Which I strongly recommend against), then you need to turn off do_resize to make the boxes workout correctly.
        #self.transform.do_resize = False

    def prepare_inputs(self, images, targets):
        # TODO move to doing the per-batch resizing using pytorch interpolate, to avoid using the detr or SSD or fasterrcnn specific prepare_inputs functions
        # https://discuss.pytorch.org/t/how-to-do-image-resizing-with-bilinear-interpolation-using-torch-tensor-on-cuda/124894
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if targets is not None:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        if self.training:
            assert targets is not None

        # create copies of the data, as we are modifying things, and we don't want the changes reflected in the caller
        images = copy.deepcopy(images)
        targets = copy.deepcopy(targets)

        # get the device the training data is on
        device = images[0].device

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        # this pre-processor needs the images on the cpu as numpy arrays
        for k in range(len(images)):
            images[k] = images[k].detach().cpu().numpy()

        if targets is not None:
            # annotations (targets) coming from the dataloader are [x1, y1, x2, y2]
            # DETR expects [x, y, w, h] so we convert here
            for ti in range(len(targets)):
                targets[ti]['boxes'] = x1y1x2y2_to_xywh(targets[ti]['boxes'])

            # create a semblance of the COCO annotation format for the pre-processor
            annotations = list()
            # {"id": 125686, "category_id": 0, "iscrowd": 0, "segmentation": [[164.81, 417.51,......167.55, 410.64]], "image_id": 242287, "area": 42061.80340000001, "bbox": [19.23, 383.18, 314.5, 244.46]},
            for ti in range(len(targets)):
                target = targets[ti]
                wrapper = dict()
                wrapper['annotations'] = list()
                for k in range(len(target['boxes'])):
                    val = dict()
                    box = target['boxes'][k]
                    label = target['labels'][k]
                    val['category_id'] = label.detach().cpu().numpy()
                    val['bbox'] = box.detach().cpu().numpy()
                    val['area'] = 0
                    val['iscrowd'] = 0
                    wrapper['annotations'].append(val)
                wrapper['image_id'] = 0  # just need a placeholder for DETR's api to be happy
                annotations.append(wrapper)

            encoded_inputs = self.transform(images, annotations=annotations, return_tensors="pt")

            # update the targets boxes based on the applied image transformations
            for k in range(len(encoded_inputs.data['labels'])):
                targets[k]['boxes'] = encoded_inputs.data['labels'][k]['boxes'].to(device)
                targets[k]['labels'] = encoded_inputs.data['labels'][k]['class_labels'].to(device)
        else:
            encoded_inputs = self.transform(images, return_tensors="pt")


        # move data back to the device
        encoded_inputs.data['pixel_values'] = encoded_inputs.data['pixel_values'].to(device)

        # # If you have multiple images in a batch (Which I strongly recommend against) then the "image_size" the post_process requires is the padded version
        # # despite the documentation saying the post_processing function needs the actual image sizes, what it really wants (to make the boxes work out correctly) is the actual image size of the images padded to handle different sizes of image within the batch.
        # _, _, h, w = encoded_inputs.data['pixel_values'].shape
        # for i in range(len(original_image_sizes)):
        #     original_image_sizes[i] = (h, w)

        images = torchvision.models.detection.image_list.ImageList(tensors=encoded_inputs.data['pixel_values'], image_sizes=None)

        return images, targets, original_image_sizes

    def postprocess_detections(self, outputs, image_sizes):
        # based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/feature_extraction_detr.py#L677

        """
        Converts the output of [`DetrForObjectDetection`] into the format expected by the COCO api. Only supports
        PyTorch.
        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            image_sizes List[Tuple]:
                List containing the size (h, w) of each image of the batch.
        """

        # convert target_sizes from list[Tuple] into
        # target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
        #    Tensor containing the size (h, w) of each image of the batch.
        target_sizes = torch.ones((len(image_sizes), 2), device=outputs.pred_boxes.device)
        for i in range(len(image_sizes)):
            target_sizes[i, 0] = image_sizes[i][0]
            target_sizes[i, 1] = image_sizes[i][1]

        detections = self.transform.post_process(outputs, target_sizes=target_sizes)
        return detections

    def basic_forward(self, images, targets):

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py#L1366

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        return_dict = self.config.use_return_dict
        outputs = self.model(images.tensors, return_dict=return_dict)

        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None

        if targets is not None:
            # First: create the matcher
            matcher = transformers.models.detr.modeling_detr.DetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = transformers.models.detr.modeling_detr.DetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            # organize the targets based on what DETR needs
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py#L1380
            #  labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            #             Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            #             following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            #             respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            #             in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
            for i in range(len(targets)):
                target = targets[i]
                device = target['boxes'].device
                target['class_labels'] = target['labels'].type(torch.LongTensor).to(device)
                target['boxes'] = target['boxes'].type(torch.FloatTensor).to(device)

            loss_dict = criterion(outputs_loss, targets)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            # apply the weight dict to the losses
            del loss_dict['cardinality_error']
            for k in weight_dict.keys():
                loss_dict[k] = loss_dict[k] * weight_dict[k]
            loss = sum(loss_dict[k] for k in loss_dict.keys() if k in weight_dict)

        detr_obj_det_outputs = transformers.models.detr.modeling_detr.DetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

        return loss_dict, detr_obj_det_outputs

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        # based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py#L1366
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images, targets, original_image_sizes = self.prepare_inputs(images, targets)

        losses, detr_obj_det_outputs = self.basic_forward(images, targets)

        detections = self.postprocess_detections(detr_obj_det_outputs, image_sizes=original_image_sizes)

        if targets is None:
            return detections
        else:
            return losses, detections