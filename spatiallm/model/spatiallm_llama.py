# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig

try:
    import torchsparse
    from torchsparse.utils.collate import sparse_collate
except ImportError:
    pass  # Ignore the import error if torchsparse is not installed for SpatialLM1.1

from spatiallm.model import (
    PointBackboneType,
    ProjectorType,
    center_crop_point_tokens,
)
from spatiallm.model.point_token_scorer import (
    score_and_hard_topk_point_tokens,
    score_and_select_point_tokens,
    score_and_threshold_point_tokens,
)
from spatiallm.model.bbox_regression import (
    BboxRegressionHead,
    apply_bbox_regression_auxiliary_loss,
)
from spatiallm.model.bbox_point_alignment import (
    BboxPointAlignmentHead,
    apply_bbox_point_alignment_auxiliary_loss,
)

IGNORE_INDEX = -100
logger = logging.get_logger(__name__)


class SpatialLMLlamaConfig(LlamaConfig):
    model_type = "spatiallm_llama"


class SpatialLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = SpatialLMLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.point_backbone_type = PointBackboneType(config.point_backbone)
        self.point_backbone = None
        point_config = config.point_config
        self.max_point_tokens = point_config.get("max_point_tokens")
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            from spatiallm.model.scenescript_encoder import PointCloudEncoder

            self.point_backbone = PointCloudEncoder(
                input_channels=point_config["input_channels"],
                d_model=point_config["embed_channels"],
                conv_layers=point_config["conv_layers"],
                num_bins=point_config["num_bins"],
            )
            embed_channels = point_config["embed_channels"]
        elif self.point_backbone_type == PointBackboneType.SONATA:
            from spatiallm.model.sonata_encoder import Sonata

            self.point_backbone = Sonata(
                in_channels=point_config["in_channels"],
                order=point_config["order"],
                stride=point_config["stride"],
                enc_depths=point_config["enc_depths"],
                enc_channels=point_config["enc_channels"],
                enc_num_head=point_config["enc_num_head"],
                enc_patch_size=point_config["enc_patch_size"],
                mlp_ratio=point_config["mlp_ratio"],
                mask_token=point_config["mask_token"],
                enc_mode=point_config["enc_mode"],
                enable_fourier_encode=True,
                num_bins=point_config["num_bins"],
                world_size=point_config.get("world_size", 32.0),
            )
            embed_channels = point_config["enc_channels"][-1]
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

        self.projector_type = ProjectorType(getattr(config, "projector", "linear"))
        if self.projector_type == ProjectorType.LINEAR:
            self.point_proj = nn.Linear(embed_channels, config.hidden_size)
        elif self.projector_type == ProjectorType.MLP:
            self.point_proj = nn.Sequential(
                nn.Linear(embed_channels, embed_channels),
                nn.GELU(),
                nn.Linear(embed_channels, config.hidden_size),
            )
        else:
            raise ValueError(f"Unknown projector type: {self.projector_type}")

        self.point_start_token_id = self.config.point_start_token_id
        self.point_end_token_id = self.config.point_end_token_id
        self.point_token_id = self.config.point_token_id

        self.bbox_regression_aux = bool(
            point_config.get("bbox_regression_aux", False)
        )
        self.bbox_regression_loss_weight = float(
            point_config.get("bbox_regression_loss_weight", 0.1)
        )
        self.bbox_regression_center_loss_weight = float(
            point_config.get("bbox_regression_center_loss_weight", 1.0)
        )
        self.bbox_regression_size_loss_weight = float(
            point_config.get("bbox_regression_size_loss_weight", 0.5)
        )
        self.bbox_regression_yaw_loss_weight = float(
            point_config.get("bbox_regression_yaw_loss_weight", 0.5)
        )
        self.bbox_regression_smooth_l1_beta = float(
            point_config.get("bbox_regression_smooth_l1_beta", 0.1)
        )
        if self.bbox_regression_aux:
            self.bbox_regression_head = BboxRegressionHead(
                input_dim=config.hidden_size,
                hidden_dim=int(
                    point_config.get("bbox_regression_hidden_dim", 1024)
                ),
                bottleneck_dim=int(
                    point_config.get("bbox_regression_bottleneck_dim", 256)
                ),
            )

        self.bbox_point_alignment_aux = bool(
            point_config.get("bbox_point_alignment_aux", False)
        )
        self.bbox_point_alignment_loss_weight = float(
            point_config.get("bbox_point_alignment_loss_weight", 0.1)
        )
        self.bbox_point_alignment_knn_k = int(
            point_config.get("bbox_point_alignment_knn_k", 8)
        )
        self.bbox_point_alignment_negative_distance_power = float(
            point_config.get(
                "bbox_point_alignment_negative_distance_power",
                2.0,
            )
        )
        self.bbox_point_alignment_detach_point_features = bool(
            point_config.get(
                "bbox_point_alignment_detach_point_features",
                True,
            )
        )
        if self.bbox_point_alignment_aux:
            if self.point_backbone_type != PointBackboneType.SONATA:
                raise NotImplementedError(
                    "BBox point alignment currently requires the Sonata backbone."
                )
            self.bbox_point_alignment_head = BboxPointAlignmentHead(
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                hidden_dim=int(
                    point_config.get("bbox_point_alignment_hidden_dim", 1024)
                ),
            )

        # Initialize weights and apply final processing
        self.post_init()

    def forward_point_cloud(
        self,
        point_cloud,
        device,
        dtype,
        point_token_keep_bboxes=None,
        return_grid_coord: bool = False,
    ):
        # point cloud has shape (n_points, n_features)
        # find the points that have nan values
        self.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            if return_grid_coord:
                raise NotImplementedError(
                    "BBox point alignment grid coordinates require Sonata."
                )
            pc_sparse_tensor = torchsparse.SparseTensor(coords=coords, feats=feats)
            pc_sparse_tensor = sparse_collate([pc_sparse_tensor])  # batch_size = 1
            pc_sparse_tensor = pc_sparse_tensor.to(device)
            encoded_features = self.point_backbone(pc_sparse_tensor)
            point_features = center_crop_point_tokens(
                encoded_features["context"],
                self.max_point_tokens,
            )
            return self.point_proj(point_features.to(dtype))
        elif self.point_backbone_type == PointBackboneType.SONATA:
            input_dict = {
                "coord": feats[:, :3].to(device),
                "grid_coord": coords.to(device),
                "feat": feats.to(device),
                "batch": torch.zeros(coords.shape[0], dtype=torch.long).to(device),
            }
            if point_token_keep_bboxes is not None:
                input_dict["point_token_keep_bboxes"] = point_token_keep_bboxes.to(device)
            if return_grid_coord:
                input_dict["return_grid_coord"] = True
            encoded_features = self.point_backbone(input_dict)
            if return_grid_coord:
                context = center_crop_point_tokens(
                    encoded_features["context"],
                    self.max_point_tokens,
                )
                grid_coord = center_crop_point_tokens(
                    encoded_features["grid_coord"],
                    self.max_point_tokens,
                )
                point_tokens = self.point_proj(context.to(dtype)).unsqueeze(0)
                return point_tokens, grid_coord
            encoded_features = center_crop_point_tokens(
                encoded_features,
                self.max_point_tokens,
            )
            # add the batch dimension
            encoded_features = encoded_features.unsqueeze(0)
            return self.point_proj(encoded_features.to(dtype))
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

    def forward_point_cloud_batch(
        self,
        point_cloud,
        offsets,
        device,
        dtype,
        point_token_keep_bboxes=None,
    ):
        """Encode a packed local batch with one Sonata forward call."""
        if self.point_backbone_type != PointBackboneType.SONATA:
            raise NotImplementedError(
                "Packed point-cloud batch encoding currently supports Sonata only."
            )
        if point_cloud.ndim != 2 or point_cloud.shape[-1] < 9:
            raise ValueError(
                "Packed point_cloud must have shape [sum_points, 9], "
                f"got {tuple(point_cloud.shape)}"
            )
        if offsets.ndim != 1 or offsets.numel() == 0:
            raise ValueError("point_cloud_offsets must be a non-empty 1D tensor.")
        coords = point_cloud[:, :3].to(device=device, dtype=torch.int64)
        feats = point_cloud[:, 3:].to(device=device, dtype=torch.float32)
        input_dict = {
            "coord": feats[:, :3],
            "grid_coord": coords,
            "feat": feats,
            "offset": offsets.to(device=device, dtype=torch.long),
            "return_grid_coord": True,
        }
        if point_token_keep_bboxes is not None:
            input_dict["point_token_keep_bboxes"] = (
                point_token_keep_bboxes.to(device)
            )
        encoded = self.point_backbone(input_dict)
        encoded["point_tokens"] = self.point_proj(encoded["context"].to(dtype))
        return encoded

    def _packed_point_features(
        self,
        point_cloud,
        point_cloud_offsets,
        point_token_keep_bboxes,
        point_token_scorer_gt_bboxes,
        device,
        dtype,
        return_grid_coord: bool = False,
    ):
        encoded = self.forward_point_cloud_batch(
            point_cloud,
            point_cloud_offsets,
            device,
            dtype,
            point_token_keep_bboxes,
        )
        scorer = getattr(self, "point_token_scorer", None)
        if bool(
            getattr(self, "point_token_scorer_frozen_threshold", False)
        ):
            if scorer is None:
                raise RuntimeError(
                    "Frozen threshold point-token selection is enabled but no "
                    "point_token_scorer is attached to the model."
                )
            max_keep = getattr(self, "point_token_scorer_max_keep", None)
            if max_keep is None:
                max_keep = (
                    self.max_point_tokens
                    or encoded["point_tokens"].shape[0]
                )
            batch_size = int(encoded["offset"].numel())
            raw_threshold = int(
                getattr(
                    self,
                    "point_token_scorer_raw_token_threshold_exclusive",
                    0,
                )
            )
            raw_token_counts = torch.bincount(
                encoded["batch"].long(),
                minlength=batch_size,
            )
            self._last_scorer_applied_region_mask = (
                raw_token_counts > raw_threshold
            )
            selected, selected_grids, scorer_metrics = (
                score_and_threshold_point_tokens(
                    scorer=scorer,
                    point_tokens=encoded["point_tokens"],
                    grid_coord=encoded["grid_coord"],
                    token_batch=encoded["batch"],
                    batch_size=batch_size,
                    threshold=float(
                        getattr(
                            self,
                            "point_token_scorer_threshold",
                            0.5,
                        )
                    ),
                    min_keep=int(
                        getattr(self, "point_token_scorer_min_keep", 1)
                    ),
                    max_keep=int(max_keep),
                    raw_token_threshold_exclusive=raw_threshold,
                )
            )
            self._last_joint_metrics.update(scorer_metrics)
            if return_grid_coord:
                return selected, selected_grids
            return selected

        if bool(getattr(self, "point_token_scorer_hard_topk", False)):
            if return_grid_coord:
                raise ValueError(
                    "BBox point alignment cannot be combined with hard-topk "
                    "point-token selection in the current implementation."
                )
            if scorer is None:
                raise RuntimeError(
                    "Hard top-k point-token selection is enabled but no "
                    "point_token_scorer is attached to the model."
                )
            selected, scorer_metrics = score_and_hard_topk_point_tokens(
                scorer=scorer,
                point_tokens=encoded["point_tokens"],
                grid_coord=encoded["grid_coord"],
                token_batch=encoded["batch"],
                batch_size=int(encoded["offset"].numel()),
                top_k=int(getattr(self, "point_token_scorer_top_k", 512)),
                max_point_tokens=self.max_point_tokens,
                detach_scorer_input=True,
            )
            self._last_joint_metrics.update(scorer_metrics)
            return selected

        if point_token_scorer_gt_bboxes is not None:
            if return_grid_coord:
                raise ValueError(
                    "BBox point alignment cannot be combined with joint scorer "
                    "selection in the current implementation."
                )
            if scorer is None:
                raise RuntimeError(
                    "point_token_scorer_gt_bboxes were provided but no "
                    "point_token_scorer is attached to the model."
                )
            max_keep = getattr(self, "point_token_scorer_max_keep", None)
            if max_keep is None:
                max_keep = self.max_point_tokens or encoded["point_tokens"].shape[0]
            selected, scorer_loss, scorer_metrics = score_and_select_point_tokens(
                scorer=scorer,
                point_tokens=encoded["point_tokens"],
                grid_coord=encoded["grid_coord"],
                token_batch=encoded["batch"],
                gt_bboxes=point_token_scorer_gt_bboxes,
                voxel_size=self.point_backbone.final_voxel_size,
                threshold=float(getattr(self, "point_token_scorer_threshold", 0.5)),
                min_keep=int(getattr(self, "point_token_scorer_min_keep", 1)),
                max_keep=int(max_keep),
                pos_weight=getattr(self, "point_token_scorer_pos_weight", None),
                detach_scorer_input=bool(
                    getattr(self, "point_token_scorer_detach_input", True)
                ),
            )
            self._joint_scorer_loss = scorer_loss
            self._last_joint_metrics.update(scorer_metrics)
            return selected

        point_features = []
        point_grids = []
        start = 0
        for end in encoded["offset"].tolist():
            features = center_crop_point_tokens(
                encoded["point_tokens"][start:end], self.max_point_tokens
            )
            grid_coord = center_crop_point_tokens(
                encoded["grid_coord"][start:end], self.max_point_tokens
            )
            point_features.append(features.unsqueeze(0))
            point_grids.append(grid_coord)
            start = end
        if return_grid_coord:
            return point_features, point_grids
        return point_features

    def set_point_backbone_dtype(self, dtype: torch.dtype):
        for param in self.point_backbone.parameters():
            param.data = param.data.to(dtype)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        point_clouds: Optional[torch.Tensor] = None,
        point_cloud_offsets: Optional[torch.Tensor] = None,
        point_token_keep_bboxes: Optional[torch.Tensor] = None,
        point_token_scorer_gt_bboxes: Optional[torch.Tensor] = None,
        point_token_features: Optional[torch.Tensor] = None,
        point_token_grid_coords: Optional[torch.Tensor] = None,
        bbox_regression_token_positions: Optional[torch.Tensor] = None,
        bbox_regression_token_mask: Optional[torch.Tensor] = None,
        bbox_regression_targets: Optional[torch.Tensor] = None,
        bbox_regression_mask: Optional[torch.Tensor] = None,
        bbox_regression_loss_scale: float = 1.0,
        bbox_regression_accumulation_divisor: float = 1.0,
        bbox_point_alignment_token_positions: Optional[torch.Tensor] = None,
        bbox_point_alignment_token_mask: Optional[torch.Tensor] = None,
        bbox_point_alignment_bboxes: Optional[torch.Tensor] = None,
        bbox_point_alignment_box_mask: Optional[torch.Tensor] = None,
        bbox_point_alignment_loss_scale: float = 1.0,
        bbox_point_alignment_accumulation_divisor: float = 1.0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            point_clouds (`torch.Tensor` of shape `(batch_size, n_points, n_features)`, *optional*):
                Point clouds to be used for the point cloud encoder.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("manycore-research/SpatialLM-Llama-1B")
        >>> tokenizer = AutoTokenizer.from_pretrained("manycore-research/SpatialLM-Llama-1B")

        >>> prompt = "<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {code_template}"
        >>> conversation = [{"role": "user", "content": prompt}]
        >>> input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(input_ids, point_clouds=point_clouds, max_length=4096)
        >>> tokenizer.batch_decode(generate_ids, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # compute point cloud embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        point_start_end_token_pos = []
        alignment_point_features = None
        alignment_point_grid_coords = None
        self._joint_scorer_loss = None
        self._last_joint_metrics = {}
        self._last_lm_loss = None
        self._last_scorer_applied_region_mask = None
        has_point_inputs = point_clouds is not None or point_token_features is not None
        if point_token_grid_coords is not None and point_token_features is None:
            raise ValueError(
                "point_token_grid_coords requires matching point_token_features."
            )
        if (
            self.point_backbone is not None
            and (input_ids.shape[1] != 1 or self.training)
            and has_point_inputs
        ):
            point_features = []
            if point_cloud_offsets is not None:
                if point_token_features is not None:
                    raise ValueError(
                        "point_cloud_offsets cannot be combined with point_token_features."
                    )
                packed_result = self._packed_point_features(
                    point_clouds,
                    point_cloud_offsets,
                    point_token_keep_bboxes,
                    point_token_scorer_gt_bboxes,
                    inputs_embeds.device,
                    inputs_embeds.dtype,
                    return_grid_coord=self.bbox_point_alignment_aux,
                )
                if self.bbox_point_alignment_aux:
                    point_features, alignment_point_grid_coords = packed_result
                    alignment_point_features = [
                        features.squeeze(0) for features in point_features
                    ]
                else:
                    point_features = packed_result
            elif point_token_features is not None:
                if self.bbox_point_alignment_aux and point_token_grid_coords is None:
                    raise ValueError(
                        "BBox point alignment with offline point_token_features "
                        "requires matching point_token_grid_coords."
                    )
                if (
                    point_token_grid_coords is not None
                    and point_token_grid_coords.shape[:2]
                    != point_token_features.shape[:2]
                ):
                    raise ValueError(
                        "point_token_grid_coords must align with "
                        "point_token_features on [batch, token], got "
                        f"features={tuple(point_token_features.shape)}, "
                        f"grid={tuple(point_token_grid_coords.shape)}."
                    )
                n_point_clouds = point_token_features.shape[0]
                for i in range(n_point_clouds):  # * iterate over batch
                    cur_point_features = point_token_features[i]
                    valid_mask = ~torch.isnan(cur_point_features).any(dim=-1)
                    cur_point_features = cur_point_features[valid_mask]
                    if self.bbox_point_alignment_aux:
                        if alignment_point_features is None:
                            alignment_point_features = []
                            alignment_point_grid_coords = []
                        alignment_point_features.append(
                            cur_point_features.to(
                                device=inputs_embeds.device,
                                dtype=inputs_embeds.dtype,
                            )
                        )
                        alignment_point_grid_coords.append(
                            point_token_grid_coords[i][valid_mask].to(
                                device=inputs_embeds.device
                            )
                        )
                    point_features.append(
                        cur_point_features.to(
                            device=inputs_embeds.device,
                            dtype=inputs_embeds.dtype,
                        ).unsqueeze(0)
                    )
            else:
                n_point_clouds = point_clouds.shape[0]
                for i in range(n_point_clouds):  # * iterate over batch
                    point_cloud = point_clouds[i]
                    cur_point_token_keep_bboxes = (
                        point_token_keep_bboxes[i]
                        if point_token_keep_bboxes is not None
                        else None
                    )
                    point_result = self.forward_point_cloud(
                        point_cloud,
                        inputs_embeds.device,
                        inputs_embeds.dtype,
                        cur_point_token_keep_bboxes,
                        return_grid_coord=self.bbox_point_alignment_aux,
                    )
                    if self.bbox_point_alignment_aux:
                        point_feature, point_grid = point_result
                        if alignment_point_features is None:
                            alignment_point_features = []
                            alignment_point_grid_coords = []
                        alignment_point_features.append(
                            point_feature.squeeze(0)
                        )
                        alignment_point_grid_coords.append(point_grid)
                    else:
                        point_feature = point_result
                    point_features.append(point_feature)

            # Insert point cloud features into the input ids
            new_input_embeds = []
            new_attention_mask = []
            cur_point_idx = 0
            max_num_tokens = 0
            for cur_input_ids, cur_input_embeds, cur_attention_mask in zip(
                input_ids, inputs_embeds, attention_mask
            ):  # * input_ids: B, L; input_embeds: B, L, C
                cur_point_features = (
                    point_features[cur_point_idx]
                    .to(device=cur_input_embeds.device)
                    .squeeze(0)
                )
                num_patches = cur_point_features.shape[0]  # * number of point tokens
                num_point_start_tokens = (
                    (cur_input_ids == self.config.point_start_token_id).sum().item()
                )
                num_point_end_tokens = (
                    (cur_input_ids == self.config.point_end_token_id).sum().item()
                )
                # currently, we only support one point start and one point end token
                assert num_point_start_tokens == num_point_end_tokens == 1, (
                    "The number of point start tokens and point end tokens should be 1, "
                    f"but got {num_point_start_tokens} and {num_point_end_tokens}."
                )
                point_start_token_pos = torch.where(
                    cur_input_ids == self.config.point_start_token_id
                )[0][0]
                point_end_token_pos = torch.where(
                    cur_input_ids == self.config.point_end_token_id
                )[0][0]
                cur_new_input_embeds = torch.cat(
                    (
                        cur_input_embeds[: point_start_token_pos + 1],
                        cur_point_features,
                        cur_input_embeds[point_end_token_pos:],
                    ),
                    dim=0,
                )
                cur_new_attention_mask = torch.cat(
                    (
                        cur_attention_mask[: point_start_token_pos + 1],
                        torch.ones(num_patches, device=cur_attention_mask.device),
                        cur_attention_mask[point_end_token_pos:],
                    ),
                    dim=0,
                )

                cur_point_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
                new_attention_mask.append(cur_new_attention_mask)
                point_start_end_token_pos.append(
                    (point_start_token_pos, num_patches, point_end_token_pos)
                )
                if cur_new_input_embeds.shape[0] > max_num_tokens:
                    max_num_tokens = cur_new_input_embeds.shape[0]
            # pad the new input embeds and attention mask to the max dimension
            for i in range(len(new_input_embeds)):
                cur_input_embeds = new_input_embeds[i]
                last_row = cur_input_embeds[-1]
                padding = last_row.repeat(max_num_tokens - cur_input_embeds.shape[0], 1)
                new_input_embeds[i] = torch.cat([cur_input_embeds, padding], dim=0)

                cur_attention_mask = new_attention_mask[i]
                new_attention_mask[i] = F.pad(
                    cur_attention_mask,
                    (0, max_num_tokens - cur_attention_mask.shape[0]),
                    value=0,
                )
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            # GenerationMixin owns the original text-only attention mask and
            # cannot observe the point-token expansion performed in this
            # forward pass. Preserve the exact expanded mask (including batch
            # padding) so cached decoding can rebuild the correct key mask.
            self._generation_expanded_attention_mask = attention_mask.detach()

            assert (
                attention_mask.shape[1] == inputs_embeds.shape[1]
            ), "The length of attention mask and inputs embeds should be the same"

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # prepare new labels
            max_num_tokens = logits.shape[1]
            if point_start_end_token_pos:
                new_labels = []
                for i in range(len(point_start_end_token_pos)):
                    cur_labels = labels[i]
                    (
                        cur_point_start_token_pos,
                        num_patches,
                        cur_point_end_token_pos,
                    ) = point_start_end_token_pos[i]
                    cur_new_labels = torch.cat(
                        (
                            cur_labels[: cur_point_start_token_pos + 1],
                            torch.full(
                                (num_patches,),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                            ),
                            cur_labels[cur_point_end_token_pos:],
                        ),
                        dim=0,
                    )
                    cur_new_labels = F.pad(
                        cur_new_labels,
                        (0, max_num_tokens - cur_new_labels.shape[0]),
                        value=IGNORE_INDEX,
                    )
                    new_labels.append(cur_new_labels)
                labels = torch.stack(new_labels, dim=0)

            assert (
                labels.shape[1] == logits.shape[1]
            ), "The length of labels and logits should be the same"

            if bool(
                getattr(
                    self,
                    "point_token_scorer_loss_only_applied_regions",
                    False,
                )
            ):
                applied_mask = self._last_scorer_applied_region_mask
                if applied_mask is None:
                    raise RuntimeError(
                        "Scorer-applied-only LM loss requires packed frozen "
                        "threshold selection to record an applied-region mask."
                    )
                applied_mask = applied_mask.to(
                    device=labels.device,
                    dtype=torch.bool,
                )
                if applied_mask.shape != labels.shape[:1]:
                    raise RuntimeError(
                        "Scorer applied-region mask does not match the LM batch: "
                        f"mask={tuple(applied_mask.shape)}, "
                        f"labels={tuple(labels.shape)}."
                    )
                labels = labels.clone()
                labels[~applied_mask] = IGNORE_INDEX
                self._last_joint_metrics[
                    "scorer_lm_loss_applied_region_ratio"
                ] = applied_mask.float().mean().detach()
                if bool(applied_mask.any()):
                    loss = self.loss_function(
                        logits=logits,
                        labels=labels,
                        vocab_size=self.config.vocab_size,
                        **loss_kwargs,
                    )
                else:
                    loss = logits.sum() * 0.0
            else:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                    **loss_kwargs,
                )
        self._last_lm_loss = loss.detach() if loss is not None else None

        if self._joint_scorer_loss is not None:
            scorer_weight = float(getattr(self, "point_token_scorer_loss_weight", 1.0))
            lm_loss = loss
            scorer_term = self._joint_scorer_loss * scorer_weight
            loss = scorer_term if lm_loss is None else lm_loss + scorer_term
            self._last_joint_metrics["lm_loss"] = (
                lm_loss.detach() if lm_loss is not None else loss.new_zeros(())
            )
            self._last_joint_metrics["joint_loss"] = loss.detach()

        loss = apply_bbox_regression_auxiliary_loss(
            model=self,
            hidden_states=hidden_states,
            loss=loss,
            point_start_end_token_pos=point_start_end_token_pos,
            numeric_token_positions=bbox_regression_token_positions,
            numeric_token_mask=bbox_regression_token_mask,
            targets=bbox_regression_targets,
            mask=bbox_regression_mask,
            loss_scale=bbox_regression_loss_scale,
            accumulation_divisor=bbox_regression_accumulation_divisor,
        )
        loss = apply_bbox_point_alignment_auxiliary_loss(
            model=self,
            hidden_states=hidden_states,
            loss=loss,
            point_start_end_token_pos=point_start_end_token_pos,
            numeric_token_positions=bbox_point_alignment_token_positions,
            numeric_token_mask=bbox_point_alignment_token_mask,
            bboxes=bbox_point_alignment_bboxes,
            bbox_mask=bbox_point_alignment_box_mask,
            point_features=alignment_point_features,
            point_grid_coords=alignment_point_grid_coords,
            loss_scale=bbox_point_alignment_loss_scale,
            accumulation_divisor=bbox_point_alignment_accumulation_divisor,
        )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                if hasattr(past_key_values, "get_seq_length"):
                    cached_length = int(past_key_values.get_seq_length())
                else:
                    cached_length = int(past_key_values[0][0].shape[-2])
                expected_length = cached_length + input_ids.shape[1]
                if attention_mask.shape[-1] < expected_length:
                    expanded_mask = getattr(
                        self,
                        "_generation_expanded_attention_mask",
                        None,
                    )
                    if (
                        expanded_mask is not None
                        and expanded_mask.shape[0] == attention_mask.shape[0]
                        and expanded_mask.shape[-1] <= expected_length
                    ):
                        attention_mask = F.pad(
                            expanded_mask.to(
                                device=attention_mask.device,
                                dtype=attention_mask.dtype,
                            ),
                            (0, expected_length - expanded_mask.shape[-1]),
                            value=1,
                        )
                    else:
                        attention_mask = F.pad(
                            attention_mask,
                            (0, expected_length - attention_mask.shape[-1]),
                            value=1,
                        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
                "point_cloud_offsets": kwargs.get("point_cloud_offsets", None),
                "point_token_keep_bboxes": kwargs.get(
                    "point_token_keep_bboxes",
                    None,
                ),
                "point_token_scorer_gt_bboxes": kwargs.get(
                    "point_token_scorer_gt_bboxes", None
                ),
                "point_token_features": kwargs.get("point_token_features", None),
                "point_token_grid_coords": kwargs.get(
                    "point_token_grid_coords",
                    None,
                ),
            }
        )
        return model_inputs


AutoConfig.register("spatiallm_llama", SpatialLMLlamaConfig)
AutoModelForCausalLM.register(SpatialLMLlamaConfig, SpatialLMLlamaForCausalLM)
