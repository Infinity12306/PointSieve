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


import math
import os
from dataclasses import dataclass
from typing import Optional

from transformers import Seq2SeqTrainingArguments


def recommended_eval_steps(max_steps: int) -> int | None:
    """Choose roughly 20 evaluations, or one per 100 optimizer steps.

    The evaluation count is rounded up to an even number so short runs still
    expose both an early and a late validation point. The final post-training
    evaluation remains in addition to the periodic evaluations.
    """
    max_steps = int(max_steps)
    if max_steps <= 0:
        return None
    target_evals = min(max_steps / 100.0, 20.0)
    eval_count = max(2, 2 * math.ceil(target_evals / 2.0))
    return max(1, math.ceil(max_steps / eval_count))


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    r"""Arguments pertaining to the trainer."""

    reset_scheduler_on_resume: bool = True
    oom_split_batch_on_cuda_oom: bool = False
    target_global_batch_size: Optional[int] = None
    save_final_checkpoint: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_copy_from_run_id: Optional[str] = None
    wandb_copy_from_project: Optional[str] = None
    wandb_copy_from_entity: Optional[str] = None
    wandb_copy_page_size: int = 1000
    auto_eval_steps: bool = True
    drop_incomplete_accumulation_window: bool = True

    def __post_init__(self):
        Seq2SeqTrainingArguments.__post_init__(self)
        if (
            self.auto_eval_steps
            and self.do_train
            and self.do_eval
            and self.eval_strategy == "steps"
            and self.max_steps > 0
        ):
            self.eval_steps = recommended_eval_steps(self.max_steps)
        if self.target_global_batch_size is not None:
            launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
            effective_global_batch_size = (
                int(self.per_device_train_batch_size)
                * int(self.gradient_accumulation_steps)
                * launched_world_size
            )
            if effective_global_batch_size != self.target_global_batch_size:
                raise ValueError(
                    "Effective global batch size guard failed: "
                    f"effective={effective_global_batch_size}, "
                    f"target={self.target_global_batch_size}, "
                    f"WORLD_SIZE={launched_world_size}."
                )
        if self.save_final_checkpoint and self.load_best_model_at_end:
            raise ValueError(
                "save_final_checkpoint cannot be combined with "
                "load_best_model_at_end because the final-step checkpoint must "
                "contain final-step model weights."
            )
        if self.wandb_copy_page_size <= 0:
            raise ValueError("wandb_copy_page_size must be positive.")
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        if self.wandb_entity:
            os.environ["WANDB_ENTITY"] = self.wandb_entity
        if self.wandb_run_name:
            os.environ["WANDB_NAME"] = self.wandb_run_name
            if not self.run_name:
                self.run_name = self.wandb_run_name
        # Local Trainer checkpoints own resume state; never reuse a native W&B run.
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
