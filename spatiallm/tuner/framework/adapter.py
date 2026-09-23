# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Set

import torch

from ..framework import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..hparams import FinetuningArguments


logger = logging.get_logger(__name__)


def _locked_eval_train(module: torch.nn.Module, mode: bool = True):
    """Ignore parent train-mode changes for a frozen module subtree."""
    del mode
    return torch.nn.Module.train(module, False)


def lock_frozen_module_in_eval(
    module: torch.nn.Module,
    module_name: str,
) -> None:
    """Keep a frozen module in eval mode without per-forward tree traversal."""
    trainable = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise ValueError(
            f"Cannot eval-lock trainable module {module_name}: {trainable[:10]}"
        )
    if not bool(getattr(module, "_spatiallm_eval_mode_locked", False)):
        module.train = MethodType(_locked_eval_train, module)
        module._spatiallm_eval_mode_locked = True
    module.eval()


def get_forbidden_modules(finetuning_args: "FinetuningArguments") -> Set[str]:
    r"""
    Freezes network modules for tuning.
    """
    forbidden_modules = set()
    if finetuning_args.train_proj_only:
        if finetuning_args.freeze_point_projector:
            raise ValueError(
                "train_proj_only and freeze_point_projector cannot both be true."
            )
        forbidden_modules.update({"point_backbone", "model", "lm_head"})
    else:
        if finetuning_args.freeze_point_tower:
            forbidden_modules.add("point_backbone")
        if finetuning_args.freeze_point_projector:
            forbidden_modules.add("point_proj")
        if finetuning_args.freeze_language_tower:
            forbidden_modules.update({"model", "lm_head"})

    return forbidden_modules


def _setup_full_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")
    if (
        finetuning_args.require_full_model_training
        and finetuning_args.require_llm_only_training
    ):
        raise ValueError(
            "require_full_model_training and require_llm_only_training "
            "cannot both be true."
        )
    forbidden_modules = get_forbidden_modules(finetuning_args)
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.float32)
            param.requires_grad_(False)

    if finetuning_args.require_llm_only_training:
        point_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                "point_backbone" in name
                or "point_proj" in name
            )
        ]
        non_point_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and "point_backbone" not in name
            and "point_proj" not in name
        ]
        if point_trainable:
            raise ValueError(
                "require_llm_only_training=true, but point modules remain "
                f"trainable: {point_trainable[:10]}"
            )
        if not non_point_trainable:
            raise ValueError(
                "require_llm_only_training=true, but no non-point parameters "
                "remain trainable."
            )

    if finetuning_args.require_full_model_training:
        parameter_groups = {
            "point_backbone": [],
            "point_proj": [],
            "non_point_model": [],
        }
        frozen_parameter_groups = {
            "point_backbone": [],
            "point_proj": [],
            "non_point_model": [],
        }
        for name, parameter in model.named_parameters():
            if "point_backbone" in name:
                group_name = "point_backbone"
            elif "point_proj" in name:
                group_name = "point_proj"
            else:
                group_name = "non_point_model"
            parameter_groups[group_name].append(name)
            if not parameter.requires_grad:
                frozen_parameter_groups[group_name].append(name)

        missing_groups = [
            group_name
            for group_name, names in parameter_groups.items()
            if not names
        ]
        frozen_groups = {
            group_name: names[:10]
            for group_name, names in frozen_parameter_groups.items()
            if names
        }
        if missing_groups or frozen_groups:
            raise ValueError(
                "require_full_model_training=true, but the complete point "
                "backbone/projector/LLM path is not trainable: "
                f"missing_groups={missing_groups}, "
                f"frozen_parameters={frozen_groups}"
            )

    # force point_backbone to have float32
    model.set_point_backbone_dtype(torch.float32)
    if finetuning_args.freeze_point_tower:
        lock_frozen_module_in_eval(model.point_backbone, "point_backbone")
    if finetuning_args.freeze_point_projector:
        lock_frozen_module_in_eval(model.point_proj, "point_proj")


def init_adapter(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support only full-parameter training for now.

    Note that the trainable parameters must be cast to float32.
    """

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16:
        logger.info_rank0(
            "Pure bf16 detected, remaining trainable params in half precision."
        )
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    _setup_full_tuning(
        model, finetuning_args, is_trainable, cast_trainable_params_to_fp32
    )

    return model
