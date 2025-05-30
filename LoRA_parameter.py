import argparse
import os
from statistics import mean

import torch
from tabulate import tabulate
import json

try:
    from safetensors import safe_open

    def load_state_dict(file_path):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}, f.metadata()

except ImportError:
    from safetensors.torch import load_file

    def load_state_dict(file_path):
        return load_file(file_path)


unet_base_names = {
    "1.5": [
        "lora_unet_down_blocks",
        "lora_unet_mid_block_attentions",
        "lora_unet_up_blocks",
    ],
    "xl": [
        "lora_unet_input_blocks",
        "lora_unet_middle_block",
        "lora_unet_output_blocks",
    ],
}
unet_block_ranges = [9, 3, 9]

unet_flux_names = [
    "lora_unet_single_blocks",
    "lora_unet_double_blocks",
]

unet_ai_toolkit_flux_names = [
    "transformer.single_transformer_blocks",
    "transformer.double_transformer_blocks",
]

unet_flux_ranges = [38, 19]


unet_sd35_names = ["lora_unet_joint_blocks"]
unet_sd35_ranges = [38]

unet_chroma_names = [
    "diffusion_model.single_blocks",
    "diffusion_model.double_blocks",
]
unet_chroma_ranges = [38, 19]

te_names = [
    "lora_te_text_model_encoder_layers",
    "lora_te1_text_model_encoder_layers",
    "lora_te2_text_model_encoder_layers",
    "lora_te3_encoder_block",
]

te_block_ranges = [12, 12, 32, 21]

lora_detect = {
    "unet": {"1.5": [False, False, False], "xl": [False, False, False]},
    "unet35": [False],
    "unet_flux": [False, False],
    "unet_aitoolkit_flux": [False, False],
    "unet_chroma": [False, False],
    "te": [False, False, False, False],
}

lora_elements = {
    "unet": [{}, {}, {}],
    "unet35": [{}],
    "unet_flux": [{}, {}],
    "unet_aitoolkit_flux": [{}, {}],
    "unet_chroma": [{}, {}],
    "te": [{}, {}, {}, {}],
}
isXL = False
isFlux = False
isAiToolkitFlux = False
isChroma = False
debug_parse = {"unet": False, "te": False}


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

    if key == "unet":
        if isXL:
            whichcal = lora_detect[key]["xl"]
            base_names = base_names["xl"]
        else:
            whichcal = lora_detect[key]["1.5"]
            base_names = base_names["1.5"]
    else:
        whichcal = lora_detect[key]

    if any(whichcal):
        for element, base_name, block_range in zip(
            lora_elements[key], base_names, block_ranges
        ):
            for i in range(block_range):
                block_name = f"{base_name}_{i}"
                if isChroma or isAiToolkitFlux:
                    block_name = f"{base_name}.{i}"
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
                    min_weight = combined_tensor.abs().min().item()
                    max_weight = combined_tensor.abs().max().item()
                    block_averages_max_and_param[block_name] = (
                        avg_weight,
                        min_weight,
                        max_weight,
                        sum(parameters),
                    )
                else:
                    block_averages_max_and_param[block_name] = (None, None, None, None)

    return block_averages_max_and_param


def legacy_count_parameters(state_dict, search_terms):
    relevant_params = {
        k: v for k, v in state_dict.items() if any(term in k for term in search_terms)
    }

    return sum(param.numel() for param in relevant_params.values())


def get_total_parameters(blocks: dict, select=""):

    total = 0

    for key, (avg, minn, maxx, parameters) in blocks.items():
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
    te_cal_seperator = {"te0": {}, "te1": {}, "te2": {}, "te3": {}}
    for i in te_cal:
        for j in range(4):
            if (
                j == 0
                and f"te" in i
                and None not in te_cal[i]
                and isXL == False
                and isFlux == False
                and isChroma == False
            ):
                te_cal_seperator[f"te0"][i] = te_cal[i]
            elif f"te{j}" in i and None not in te_cal[i]:
                te_cal_seperator[f"te{j}"][i] = te_cal[i]
    return te_cal_seperator


def seperated_data(state_dict: dict):
    global isXL, isFlux, isChroma, isAiToolkitFlux
    for part, value in state_dict.items():
        for k, v in unet_base_names.items():
            for idx, val in enumerate(v):
                if val in part:
                    if k == "xl":
                        isXL = True
                    else:
                        isXL = False
                    lora_detect["unet"][k][idx] = True
                    lora_elements["unet"][idx][part] = value
        for idx, val in enumerate(unet_sd35_names):
            if val in part:
                lora_detect["unet35"][idx] = True
                lora_elements["unet35"][idx][part] = value
        for idx, val in enumerate(unet_flux_names):
            if val in part:
                lora_detect["unet_flux"][idx] = True
                lora_elements["unet_flux"][idx][part] = value
                isFlux = True
        for idx, val in enumerate(unet_ai_toolkit_flux_names):
            if val in part:
                lora_detect["unet_aitoolkit_flux"][idx] = True
                lora_elements["unet_aitoolkit_flux"][idx][part] = value
                isAiToolkitFlux = True
        for idx, val in enumerate(unet_chroma_names):
            if val in part:
                lora_detect["unet_chroma"][idx] = True
                lora_elements["unet_chroma"][idx][part] = value
                isChroma = True
        for idx, val in enumerate(te_names):
            if val in part:
                lora_detect["te"][idx] = True
                lora_elements["te"][idx][part] = value


def print_calculated(name: str, opt: dict):
    if len(opt) > 0:
        print(f"\n{name} block averages, max weights and parameters:")

        table = [
            ["block name", "average weight", "min weight", "max weight", "parameters"]
        ]

        for block, v in opt.items():

            if isAiToolkitFlux:
                block = block[12:]

            if None not in v:
                avg, min_val, max_val, parameters = v

                table.append([block, avg, min_val, max_val, f"{parameters:,}"])
            else:
                not_detected = None
                table.append(
                    [block, not_detected, not_detected, not_detected, not_detected]
                )
        print(
            tabulate(
                table,
                headers="firstrow",
                floatfmt=".16f",
                colalign=("center", "center", "center", "center", "center"),
                tablefmt="psql",
                missingval="-",
            )
        )
        print(
            f"{name} average weight : {mean(list(map(lambda x : x[0],filter(lambda x : None not in x,opt.values()))))}"
        )


def print_metadata(metadata: dict):
    print("\nMetadata")
    metadata_table = [["key", "value"]]
    if metadata:
        for k, v in sorted(metadata.items()):
            val = str(v)
            metadata_table.append([k, val if len(val) < 80 else val[:77] + "..."])
    print(
        tabulate(
            metadata_table,
            headers="firstrow",
            tablefmt="psql",
            missingval="-",
        )
    )


def main(args):
    lora_model_path = args.input
    debug = args.debug

    filename = os.path.split(lora_model_path)[-1]

    if not debug:
        debug_parse["unet"] = True
        debug_parse["te"] = True
    else:
        if "unet" in debug:
            debug_parse["unet"] = True
        if "te" in debug:
            debug_parse["te"] = True

    state_dict, metadata = load_state_dict(lora_model_path)

    if args.save_metadata:
        if metadata:
            with open(
                os.path.join("metadata_output", f"{filename}_raw_metadata.json"), "+w"
            ) as fp:
                json.dump(metadata, fp, indent=4)
        else:
            print("No metadata found")

    seperated_data(state_dict)

    if debug_parse["unet"]:
        unet_cal = calculate_parameters_avg_and_max_weights(
            "unet", unet_block_ranges, unet_base_names
        )
        unet_sd35_cal = calculate_parameters_avg_and_max_weights(
            "unet35", unet_sd35_ranges, unet_sd35_names
        )
        unet_flux_cal = calculate_parameters_avg_and_max_weights(
            "unet_flux", unet_flux_ranges, unet_flux_names
        )
        unet_ai_toolkit_flux_cal = calculate_parameters_avg_and_max_weights(
            "unet_aitoolkit_flux", unet_flux_ranges, unet_ai_toolkit_flux_names
        )
        unet_chroma_cal = calculate_parameters_avg_and_max_weights(
            "unet_chroma", unet_chroma_ranges, unet_chroma_names
        )
    if debug_parse["te"]:
        te_cal = calculate_parameters_avg_and_max_weights(
            "te", te_block_ranges, te_names
        )
        te_cal_seperated = te_sepearator(te_cal)

    if debug_parse["unet"]:
        print(
            f"UNet                                : {format_parameters(get_total_parameters(unet_cal))}"
        )
        print(
            f"Conv layer UNet                     : {format_parameters(legacy_count_parameters(state_dict, ['conv']))}"
        )
        print(
            f"UNet Joint [SD3.5]                  : {format_parameters(get_total_parameters(unet_sd35_cal))}"
        )
        print(
            f"UNet single block [Flux]            : {format_parameters(get_total_parameters(unet_flux_cal, 'single'))}"
        )
        print(
            f"UNet double block [Flux]            : {format_parameters(get_total_parameters(unet_flux_cal, 'double'))}"
        )
        print(
            f"UNet single block [AI Toolkit Flux] : {format_parameters(get_total_parameters(unet_ai_toolkit_flux_cal, 'single'))}"
        )
        print(
            f"UNet double block [AI Toolkit Flux] : {format_parameters(get_total_parameters(unet_ai_toolkit_flux_cal, 'double'))}"
        )
        print(
            f"UNet single block [Chroma]          : {format_parameters(get_total_parameters(unet_chroma_cal, 'single'))}"
        )
        print(
            f"UNet double block [Chroma]          : {format_parameters(get_total_parameters(unet_chroma_cal, 'double'))}"
        )
    if debug_parse["te"]:
        print(
            f"Text-Encoder 0 Clip_L               : {format_parameters(get_total_parameters(te_cal, 'te0')) if isFlux == False and isXL == False and isChroma == False else 'Not Detect'}"
        )
        print(
            f"Text-Encoder 1 Clip_L               : {format_parameters(get_total_parameters(te_cal, 'te1'))}"
        )
        print(
            f"Text-Encoder 2 Clip_G               : {format_parameters(get_total_parameters(te_cal, 'te2'))}"
        )
        print(
            f"Text-Encoder 3 T5XXL                : {format_parameters(get_total_parameters(te_cal, 'te3'))}"
        )

    if args.metadata:
        print_metadata(metadata)
    if debug_parse["unet"]:
        print_calculated("UNet", unet_cal)
        print_calculated("UNet SD3.5", unet_sd35_cal)
        print_calculated("UNet Flux", unet_flux_cal)
        print_calculated("UNet Flux AI Toolkit", unet_ai_toolkit_flux_cal)
        print_calculated("UNet Chroma", unet_chroma_cal)
    if debug_parse["te"]:
        print_calculated("Text-Encoder TE0", te_cal_seperated["te0"])
        print_calculated("Text-Encoder TE1", te_cal_seperated["te1"])
        print_calculated("Text-Encoder TE2", te_cal_seperated["te2"])
        print_calculated("Text-Encoder TE3", te_cal_seperated["te3"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate parameters of LoRA components."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the .safetensors file."
    )
    parser.add_argument(
        "--debug",
        nargs="+",
        help="debug specific module accept args is unet and te example --debug unet --debug te --debug unet te",
    )
    parser.add_argument(
        "--save_metadata", action="store_true", help="Saving Metadata into json file"
    )
    parser.add_argument("--metadata", action="store_true", help="Show metadata")
    args = parser.parse_args()

    main(args)
