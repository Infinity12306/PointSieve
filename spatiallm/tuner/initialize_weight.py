import argparse

import torch
import spatiallm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize SpatialLM weight")
    parser.add_argument(
        "--spatiallm_weight",
        type=str,
        default="manycore-research/SpatialLM1.1-Qwen-0.5B",
        help="Path to the SpatialLM weight",
    )
    parser.add_argument(
        "--llm_weight",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="""
            Path to the LLM weight
            Qwen/Qwen2.5-0.5B-Instruct for SpatialLM Qwen variant
            meta-llama/Llama-3.2-1B-Instruct for SpatialLM Llama variant
        """,
    )
    parser.add_argument(
        "--encoder_weight",
        type=str,
        required=True,
        help="""
            Path to the point cloud encoder weight file
            sonata.pth for SpatialLM1.1 or scenescript_model_ase.ckpt for SpatialLM1.0
        """,
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output weight folder"
    )
    args = parser.parse_args()

    print("Loading config...")
    config = AutoConfig.from_pretrained(args.spatiallm_weight)
    spatiallm_model = AutoModelForCausalLM.from_config(config)

    # load original LLM model weight
    print("Loading LLM weight...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_weight, torch_dtype="auto"
    )
    llm_state_dict = llm_model.state_dict()
    spatiallm_model.load_state_dict(llm_state_dict, strict=False)

    print("Loading encoder weight...")
    if config.point_backbone == "sonata":
        encoder_state_dict = torch.load(args.encoder_weight, weights_only=False)[
            "state_dict"
        ]
        for name, param in spatiallm_model.point_backbone.named_parameters():
            if (
                name in encoder_state_dict
                and param.shape != encoder_state_dict[name].shape
            ):
                new_param = encoder_state_dict[name][:, : param.shape[1]]
                encoder_state_dict[name] = new_param
    elif config.point_backbone == "scenescript":
        ckpt_dict = torch.load(args.encoder_weight, weights_only=False)
        encoder_state_dict = {}
        for k, v in ckpt_dict["model_state_dict"].items():
            if k.startswith("encoder"):
                encoder_state_dict[k.replace("encoder.", "")] = v
        for name, param in spatiallm_model.point_backbone.named_parameters():
            if (
                name in encoder_state_dict
                and param.shape != encoder_state_dict[name].shape
            ):
                # from [343, 3, 16] to [343, 6, 16]
                # expand the parameter to the same shape
                pretrained_param = encoder_state_dict[name]
                missing_channels = param.shape[1] - pretrained_param.shape[1]
                random_tensor = torch.randn(
                    param.shape[0], missing_channels, param.shape[2]
                )
                new_param = torch.cat([pretrained_param, random_tensor], dim=1)
                encoder_state_dict[name] = new_param
    spatiallm_model.point_backbone.load_state_dict(encoder_state_dict, strict=False)

    # save the model
    print("Saving SpatialLM weight...")
    tokenizer = AutoTokenizer.from_pretrained(args.spatiallm_weight)
    tokenizer.save_pretrained(args.output_dir)
    spatiallm_model.save_pretrained(args.output_dir)
