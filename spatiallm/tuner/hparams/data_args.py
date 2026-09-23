# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which template to use for constructing prompts in training and inference."
        },
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."
        },
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."
        },
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    dataset_info_file: str = field(
        default="dataset_info.json",
        metadata={
            "help": (
                "Dataset registry filename relative to `dataset_dir`. This can "
                "be changed when an experiment requires dated metadata files."
            )
        },
    )
    media_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."
        },
    )
    cutoff_len: int = field(
        default=8192,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each dataset."
        },
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."
        },
    )
    val_size: float = field(
        default=0.0,
        metadata={
            "help": "Size of the validation set, should be an integer or a float in range `[0,1)`."
        },
    )
    eval_on_each_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to evaluate on each dataset separately."},
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the preprocessed datasets. "
                "If save_dir not exists, it will save the preprocessed datasets. "
                "If save_dir exists, it will load the preprocessed datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use a shared file system for the datasets."
        },
    )
    num_bins: int = field(
        default=1280,
        metadata={"help": "The number of bins for point cloud quantization."},
    )
    world_size: float = field(
        default=32.0,
        metadata={
            "help": (
                "World extent in meters used for point cloud quantization and "
                "layout position discretization. Region point clouds larger than "
                "this extent on any axis are center-cropped before encoding."
            )
        },
    )
    max_point_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of encoded point tokens inserted into the "
                "language model. Longer point-token sequences are center-cropped "
                "by removing tokens from both ends."
            )
        },
    )
    point_token_bbox_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep only final point tokens whose voxels overlap "
                "GT object bboxes. This is intended for hierarchical stage-2 "
                "experiments."
            )
        },
    )
    point_token_bbox_expand_ratio: float = field(
        default=0.1,
        metadata={
            "help": (
                "Per-side expansion ratio for GT object bboxes used by "
                "point-token masking. 0.1 makes each dimension 1.2x."
            )
        },
    )
    point_cloud_batch_encoding: bool = field(
        default=False,
        metadata={
            "help": (
                "Pack all point clouds in a local batch and run the Sonata "
                "encoder once using offsets instead of per-sample encoding."
            )
        },
    )
    point_token_scorer_gt_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Return augmented GT object bboxes for online point-token "
                "scorer supervision without applying GT filtering."
            )
        },
    )
    bbox_regression_aux: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable continuous bbox regression from the mean final-layer "
                "LLM hidden state of all seven numeric bbox fields."
            )
        },
    )
    bbox_regression_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight of the continuous bbox auxiliary loss."},
    )
    bbox_regression_loss_warmup_ratio: float = field(
        default=0.05,
        metadata={
            "help": (
                "Fraction of total optimizer steps used to linearly warm up the "
                "continuous bbox auxiliary loss weight."
            )
        },
    )
    bbox_regression_center_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Inner weight of normalized bbox-center Smooth L1 loss."},
    )
    bbox_regression_size_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Inner weight of log-size Smooth L1 loss."},
    )
    bbox_regression_yaw_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Inner weight of doubled-angle yaw Smooth L1 loss."},
    )
    bbox_regression_smooth_l1_beta: float = field(
        default=0.1,
        metadata={"help": "Beta parameter for bbox Smooth L1 loss components."},
    )
    bbox_regression_hidden_dim: int = field(
        default=1024,
        metadata={"help": "First hidden dimension of the bbox regression MLP."},
    )
    bbox_regression_bottleneck_dim: int = field(
        default=256,
        metadata={"help": "Bottleneck dimension of the bbox regression MLP."},
    )
    bbox_point_alignment_aux: bool = field(
        default=False,
        metadata={
            "help": (
                "Align the mean final-layer hidden state of the seven bbox "
                "numeric attributes with projected point tokens near GT centers."
            )
        },
    )
    bbox_point_alignment_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Outer weight of the bbox/point alignment loss."},
    )
    bbox_point_alignment_loss_warmup_ratio: float = field(
        default=0.05,
        metadata={
            "help": (
                "Fraction of optimizer steps used to linearly warm up the "
                "bbox/point alignment loss weight."
            )
        },
    )
    bbox_point_alignment_hidden_dim: int = field(
        default=1024,
        metadata={"help": "Hidden dimension of the one-way bbox alignment MLP."},
    )
    bbox_point_alignment_knn_k: int = field(
        default=8,
        metadata={"help": "Number of nearest projected point tokens per GT center."},
    )
    bbox_point_alignment_negative_distance_power: float = field(
        default=2.0,
        metadata={
            "help": (
                "Exponent used before normalizing non-overlapping negative "
                "bbox-center distances into per-bbox weights."
            )
        },
    )
    bbox_point_alignment_detach_point_features: bool = field(
        default=True,
        metadata={
            "help": (
                "Detach projected point-token targets in the alignment loss so "
                "the auxiliary objective primarily shapes LLM bbox hidden states."
            )
        },
    )
    do_augmentation: bool = field(
        default=False,
        metadata={"help": "Whether or not to do data augmentation."},
    )
    random_rotation: bool = field(
        default=False,
        metadata={"help": "Whether or not to do non axis-aligned random rotation."},
    )
    geometric_augmentation: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to apply the SpatialLM geometric rotation and scale "
                "augmentation. Disable this for QA tasks whose directional "
                "language must remain aligned with the original scene."
            )
        },
    )
    eval_do_augmentation: Optional[bool] = field(
        default=None,
        metadata={"help": "Independent eval augmentation; None preserves the training setting."},
    )
    eval_geometric_augmentation: Optional[bool] = field(
        default=None,
        metadata={"help": "Independent eval rotation/scale augmentation; None inherits training."},
    )
    eval_point_sampling: Optional[str] = field(
        default=None,
        metadata={"help": "Eval GridSample mode: test is deterministic; train/None preserves random voxel sampling."},
    )
    bbox_filter_inverse_rotation: bool = field(
        default=False,
        metadata={"help": "Use the correct world-to-box inverse rotation for the >100-point GT filter; False preserves historical supervision."},
    )

    def __post_init__(self):
        for name in ("eval_do_augmentation", "eval_geometric_augmentation"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError(f"`{name}` must be a YAML boolean or null.")
        if self.eval_point_sampling not in (None, "train", "test"):
            raise ValueError("`eval_point_sampling` must be null, train or test.")
        if type(self.bbox_filter_inverse_rotation) is not bool:
            raise ValueError("`bbox_filter_inverse_rotation` must be a YAML boolean.")
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

        if self.media_dir is None:
            self.media_dir = self.dataset_dir

        if not self.dataset_info_file.strip():
            raise ValueError("`dataset_info_file` must be non-empty.")

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")
        if self.world_size <= 0:
            raise ValueError("`world_size` must be positive.")
        if self.max_point_tokens is not None and self.max_point_tokens <= 0:
            raise ValueError("`max_point_tokens` must be positive when configured.")
        if self.point_token_bbox_expand_ratio < 0:
            raise ValueError("`point_token_bbox_expand_ratio` must be non-negative.")
        if self.bbox_regression_loss_weight < 0:
            raise ValueError("`bbox_regression_loss_weight` must be non-negative.")
        if not 0 <= self.bbox_regression_loss_warmup_ratio <= 1:
            raise ValueError(
                "`bbox_regression_loss_warmup_ratio` must be in [0, 1]."
            )
        for name in (
            "bbox_regression_center_loss_weight",
            "bbox_regression_size_loss_weight",
            "bbox_regression_yaw_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must be non-negative.")
        if self.bbox_regression_smooth_l1_beta <= 0:
            raise ValueError("`bbox_regression_smooth_l1_beta` must be positive.")
        if self.bbox_regression_hidden_dim <= 0:
            raise ValueError("`bbox_regression_hidden_dim` must be positive.")
        if self.bbox_regression_bottleneck_dim <= 0:
            raise ValueError("`bbox_regression_bottleneck_dim` must be positive.")
        if self.bbox_point_alignment_loss_weight < 0:
            raise ValueError(
                "`bbox_point_alignment_loss_weight` must be non-negative."
            )
        if not 0 <= self.bbox_point_alignment_loss_warmup_ratio <= 1:
            raise ValueError(
                "`bbox_point_alignment_loss_warmup_ratio` must be in [0, 1]."
            )
        if self.bbox_point_alignment_hidden_dim <= 0:
            raise ValueError(
                "`bbox_point_alignment_hidden_dim` must be positive."
            )
        if self.bbox_point_alignment_knn_k <= 0:
            raise ValueError("`bbox_point_alignment_knn_k` must be positive.")
        if self.bbox_point_alignment_negative_distance_power <= 0:
            raise ValueError(
                "`bbox_point_alignment_negative_distance_power` must be positive."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
