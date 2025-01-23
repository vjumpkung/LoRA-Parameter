import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import os
from statistics import mean
from PIL import Image


try:
    from safetensors import safe_open

    def load_state_dict(file_path):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}

except ImportError:
    from safetensors.torch import load_file

    def load_state_dict(file_path):
        return load_file(file_path)


unet_base_names = [
    "lora_unet_input_blocks",
    "lora_unet_middle_block",
    "lora_unet_output_blocks",
]
unet_block_ranges = [9, 3, 9]

unet_flux_names = [
    "lora_unet_single_blocks",
    "lora_unet_double_blocks",
]
unet_flux_ranges = [38, 19]
te_names = [
    "lora_te1_text_model_encoder_layers",
    "lora_te2_text_model_encoder_layers",
    "lora_te3_text_model_encoder_layers",
]

te_block_ranges = [12, 32, 24]

lora_detect = {
    "unet": [False, False, False],
    "unet_flux": [False, False],
    "te": [False, False, False],
}

lora_elements = {
    "unet": [{}, {}, {}],
    "unet_flux": [{}, {}],
    "te": [{}, {}, {}],
}


def format_parameters(param_count):
    if param_count == 0:
        return "Not Detect"
    if param_count >= 1_000_000:
        formatted = f"{param_count / 1_000_000:.2f}M parameters"
    elif param_count >= 1_000:
        formatted = f"{param_count / 1_000:.2f}K parameters"
    else:
        formatted = f"{param_count} parameters"

    full_count_with_commas = f"{param_count:,}"
    return f"{formatted} ({full_count_with_commas})"


def calculate_parameters_avg_and_max_weights(key, block_ranges, base_names):
    block_averages_max_and_param = {}
    if any(lora_detect[key]):
        for element, base_name, block_range in zip(
            lora_elements[key], base_names, block_ranges
        ):
            for i in range(block_range):
                block_name = f"{base_name}_{i}"
                block_weights = []
                parameters = []
                for k, v in element.items():
                    if (
                        block_name in k
                        and "alpha" not in k
                        and isinstance(v, torch.Tensor)
                    ):
                        block_weights.append(v.to(torch.float32).flatten())
                    if block_name in k:
                        parameters.append(v.numel())
                if block_weights:
                    combined_tensor = torch.cat(block_weights)
                    avg_weight = combined_tensor.abs().mean().item()
                    max_weight = combined_tensor.abs().max().item()
                    block_averages_max_and_param[block_name] = (
                        avg_weight,
                        max_weight,
                        sum(parameters),
                    )
                else:
                    block_averages_max_and_param[block_name] = (None, None, None)

    return block_averages_max_and_param


def legacy_count_parameters(state_dict, search_terms):
    relevant_params = {
        k: v for k, v in state_dict.items() if any(term in k for term in search_terms)
    }

    return sum(param.numel() for param in relevant_params.values())


def get_total_parameters(blocks: dict, select=""):

    total = 0

    for key, (avg, maxx, parameters) in blocks.items():
        if (
            select not in ["single", "double", "te1", "te2", "te3"]
            and parameters is not None
        ):
            total += parameters
        else:
            if select in key and parameters is not None:
                total += parameters

    return total


def te_sepearator(te_cal: dict):
    te_cal_seperator = {"te1": {}, "te2": {}, "te3": {}}
    for i in te_cal:
        for j in [1, 2, 3]:
            if f"te{j}" in i and None not in te_cal[i]:
                te_cal_seperator[f"te{j}"][i] = te_cal[i]
    return te_cal_seperator


def seperated_data(state_dict: dict):
    for part, value in state_dict.items():
        for idx, val in enumerate(unet_base_names):
            if val in part:
                lora_detect["unet"][idx] = True
                lora_elements["unet"][idx][part] = value
        for idx, val in enumerate(unet_flux_names):
            if val in part:
                lora_detect["unet_flux"][idx] = True
                lora_elements["unet_flux"][idx][part] = value
        for idx, val in enumerate(te_names):
            if val in part:
                lora_detect["te"][idx] = True
                lora_elements["te"][idx][part] = value


def print_calculated(name: str, opt: dict):
    if len(opt) > 0:
        print(f"\n{name} block averages, max weights and parameters:")
        for block, v in opt.items():
            if None not in v:
                avg, max_val, parameters = v
                print(
                    f"{block} average weight: {avg:.16f}, max weight: {max_val:.16f} num params: {parameters:,}"
                )
            else:
                print(f"{block} Not Detect")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate parameters of LoRA components."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the .safetensors file."
    )
    args = parser.parse_args()

    lora_model_path = args.input
    state_dict = load_state_dict(lora_model_path)

    seperated_data(state_dict)

    unet_cal = calculate_parameters_avg_and_max_weights(
        "unet", unet_block_ranges, unet_base_names
    )
    unet_flux_cal = calculate_parameters_avg_and_max_weights(
        "unet_flux", unet_flux_ranges, unet_flux_names
    )
    te_cal = calculate_parameters_avg_and_max_weights("te", te_block_ranges, te_names)
    te_cal_seperated = te_sepearator(te_cal)

    print(
        f"UNet                     : {format_parameters(get_total_parameters(unet_cal))}"
    )
    print(
        f"Conv layer UNet          : {format_parameters(legacy_count_parameters(state_dict, ['conv']))}"
    )
    print(
        f"UNet single block [Flux] : {format_parameters(get_total_parameters(unet_flux_cal, 'single'))}"
    )
    print(
        f"UNet double block [Flux] : {format_parameters(get_total_parameters(unet_flux_cal, 'double'))}"
    )
    print(
        f"Text-Encoder 1 Clip_L    : {format_parameters(get_total_parameters(te_cal, 'te1'))}"
    )
    print(
        f"Text-Encoder 2 Clip_G    : {format_parameters(get_total_parameters(te_cal, 'te2'))}"
    )
    print(
        f"Text-Encoder 3 T5XXL     : {format_parameters(get_total_parameters(te_cal, 'te3'))}"
    )

    print_calculated("UNet", unet_cal)
    print_calculated("UNet Flux", unet_flux_cal)
    print_calculated("Text-Encoder TE1", te_cal_seperated["te1"])
    print_calculated("Text-Encoder TE2", te_cal_seperated["te2"])
    print_calculated("Text-Encoder TE3", te_cal_seperated["te3"])


if __name__ == "__main__":
    main()
