import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import os
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

unet_flux_name = [
    "lora_unet_single_blocks",
    "lora_unet_double_blocks",
]
unet_flux_ranges = [38, 19]

unet_plot_name = {
    "lora_unet_input_blocks": "IN",
    "lora_unet_middle_block": "MID",
    "lora_unet_output_blocks": "OUT",
    "lora_unet_single_blocks": "SINGLE_",
    "lora_unet_double_blocks": "DOUBLE_",
}

te_plot_name = {
    "lora_te1_layers": "TE1_",
    "lora_te2_layers": "TE2_",
    "lora_te3_layers": "TE3_",
}


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], f"{y[i]*100:.4f}", ha="center", va="bottom")


def count_parameters(state_dict, search_terms):
    relevant_params = {
        k: v for k, v in state_dict.items() if any(term in k for term in search_terms)
    }
    return sum(param.numel() for param in relevant_params.values()), relevant_params


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


def count_for_plot(*args):
    total_detected = 0
    for i in args:
        if i:
            total_detected += 1
    return total_detected


def get_weight_vector_and_average_by_block(state_dict, base_names, block_ranges):
    block_averages_and_max = {}

    for base_name, block_range in zip(base_names, block_ranges):
        for i in range(block_range):
            block_name = f"{base_name}_{i}"
            block_weights = []

            for k, v in state_dict.items():
                if block_name in k and "alpha" not in k and isinstance(v, torch.Tensor):
                    block_weights.append(v.to(torch.float32).flatten())

            if block_weights:
                combined_tensor = torch.cat(block_weights)
                avg_weight = combined_tensor.abs().mean().item()
                max_weight = combined_tensor.abs().max().item()
                block_averages_and_max[block_name] = (avg_weight, max_weight)
            else:
                block_averages_and_max[block_name] = ("Not Detect", "Not Detect")

    return block_averages_and_max


def main():
    parser = argparse.ArgumentParser(
        description="Calculate parameters of LoRA components."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the .safetensors file."
    )
    args = parser.parse_args()

    lora_model_path = args.input

    filename = os.path.split(lora_model_path)[-1]

    state_dict = load_state_dict(lora_model_path)

    unet_params, _ = count_parameters(state_dict, ["lora_unet"])
    conv_params, _ = count_parameters(state_dict, ["conv"])
    unet_single_params, _ = count_parameters(state_dict, ["lora_unet_single"])
    unet_double_params, _ = count_parameters(state_dict, ["lora_unet_double"])
    text_encoder_1_params, _ = count_parameters(
        state_dict, ["lora_te_text_model_encoder", "lora_te1_text_model_encoder"]
    )
    text_encoder_2_params, _ = count_parameters(state_dict, ["lora_te2"])
    text_encoder_3_params, _ = count_parameters(state_dict, ["lora_te3"])

    print(f"UNet                    : {format_parameters(unet_params)}")
    print(f"Conv layer UNet         : {format_parameters(conv_params)}")
    print(f"UNet single block [Flux]: {format_parameters(unet_single_params)}")
    print(f"UNet double block [Flux]: {format_parameters(unet_double_params)}")
    print(f"Text-Encoder 1 Clip_L   : {format_parameters(text_encoder_1_params)}")
    print(f"Text-Encoder 2 Clip_G   : {format_parameters(text_encoder_2_params)}")
    print(f"Text-Encoder 3 T5       : {format_parameters(text_encoder_3_params)}")

    row_counts = count_for_plot(
        unet_params,
        unet_single_params + unet_double_params,
        text_encoder_1_params,
        text_encoder_2_params,
        text_encoder_3_params,
    )

    if unet_params and not (unet_single_params + unet_double_params):
        row_counts += 1

    if not (unet_single_params + unet_double_params):
        plt.figure(figsize=(30, 10 * row_counts))
    else:
        plt.figure(figsize=(50, 10 * row_counts))

    plt.rcParams.update({"font.size": 14})

    count = 1

    if any(name in key for key in state_dict.keys() for name in unet_base_names):

        unet_x_name = []
        unet_y_avg_value = []
        unet_y_max_value = []

        unet_block_averages_and_max = get_weight_vector_and_average_by_block(
            state_dict, unet_base_names, unet_block_ranges
        )
        print("\nUNet block averages and max weights:")
        for block, (avg, max_val) in unet_block_averages_and_max.items():
            if isinstance(avg, str) or isinstance(max_val, str):
                print(f"{block} average weight: {avg}, max weight: {max_val}")
            else:
                print(f"{block} average weight: {avg:.16f}, max weight: {max_val:.16f}")
                unet_y_avg_value.append(round(avg, 16))
                unet_y_max_value.append(round(max_val, 16))
                unet_match_pattern = re.split(r"(.*?)_(\d+)$", block)
                unet_x_name.append(
                    unet_plot_name[unet_match_pattern[1]] + unet_match_pattern[2]
                )

        plt.subplot(row_counts, 1, count)
        plt.plot(unet_x_name, unet_y_avg_value, marker="o")
        addlabels(unet_x_name, unet_y_avg_value)
        plt.title("UNet block averages")
        count += 1
        plt.subplot(row_counts, 1, count)
        plt.plot(unet_x_name, unet_y_max_value, marker="o")
        addlabels(unet_x_name, unet_y_max_value)
        plt.title("UNet block max weights")
        count += 1

    if any(name in key for key in state_dict.keys() for name in unet_flux_name):
        unet_fluxblock_averages = get_weight_vector_and_average_by_block(
            state_dict, unet_flux_name, unet_flux_ranges
        )

        unet_flux_x_name = []
        unet_flux_y_avg1_value = []
        unet_flux_y_max_value = []

        print("\nUNet Flux block averages:")
        for block, avg in unet_fluxblock_averages.items():
            if isinstance(avg, str):
                print(f"{block} average weight: {avg}")
            else:
                print(
                    f"{block} average weight: {avg[0]:.16f}, max weight: {avg[1]:.16f}"
                )
                unet_flux_y_avg1_value.append(round(avg[0], 16))
                unet_flux_y_max_value.append(round(avg[1], 16))
                unet_match_pattern = re.split(r"(.*?)_(\d+)$", block)
                unet_flux_x_name.append(
                    unet_plot_name[unet_match_pattern[1]] + unet_match_pattern[2]
                )

        plt.subplot(row_counts, 1, count)
        plt.plot(unet_flux_x_name, unet_flux_y_avg1_value, marker="o")
        plt.xticks(rotation=90)
        addlabels(unet_flux_x_name, unet_flux_y_avg1_value)
        plt.title("UNet Flux block averages")
        count += 1

        plt.subplot(row_counts, 1, count)
        plt.plot(unet_flux_x_name, unet_flux_y_max_value, marker="o")
        plt.xticks(rotation=90)
        addlabels(unet_flux_x_name, unet_flux_y_max_value)
        plt.title("UNet Flux block max")
        count += 1

    if any("lora_te1_text_model_encoder_layers" in key for key in state_dict.keys()):
        text_encoder_te1_layer_averages_and_max = (
            get_weight_vector_and_average_by_block(
                state_dict, ["lora_te1_text_model_encoder_layers"], [12]
            )
        )

        te1_x_name = []
        te1_avg = []

        print("\nText-Encoder TE1 layers average weights:")
        for layer, (avg, _) in text_encoder_te1_layer_averages_and_max.items():
            short_layer_name = layer.replace(
                "lora_te1_text_model_encoder_", "lora_te1_"
            )
            if isinstance(avg, str):
                print(f"{short_layer_name} average weight: {avg}")
            else:
                print(f"{short_layer_name} average weight: {avg:.16f}")
                te1_avg.append(round(avg, 16))
                te1_pattern = re.split(r"(.*?)_(\d+)$", short_layer_name)
                te1_x_name.append(te_plot_name[te1_pattern[1]] + te1_pattern[2])

        plt.subplot(row_counts, 1, count)
        plt.plot(te1_x_name, te1_avg, marker="o")
        addlabels(te1_x_name, te1_avg)
        plt.title("Text-Encoder TE1 layers average weights")
        count += 1

    if any("lora_te2_text_model_encoder_layers" in key for key in state_dict.keys()):
        text_encoder_te1_layer_averages_and_max = (
            get_weight_vector_and_average_by_block(
                state_dict, ["lora_te2_text_model_encoder_layers"], [32]
            )
        )

        te2_x_name = []
        te2_avg = []

        print("\nText-Encoder TE2 layers average weights:")
        for layer, (avg, _) in text_encoder_te1_layer_averages_and_max.items():
            short_layer_name = layer.replace(
                "lora_te2_text_model_encoder_", "lora_te2_"
            )
            if isinstance(avg, str):
                print(f"{short_layer_name} average weight: {avg}")
            else:
                print(f"{short_layer_name} average weight: {avg:.16f}")
                te2_avg.append(round(avg, 16))
                te2_pattern = re.split(r"(.*?)_(\d+)$", short_layer_name)
                te2_x_name.append(te_plot_name[te2_pattern[1]] + te2_pattern[2])

        plt.subplot(row_counts, 1, count)
        plt.plot(te2_x_name, te2_avg, marker="o")
        plt.xticks(rotation=45)
        addlabels(te2_x_name, te2_avg)
        plt.title("Text-Encoder TE2 layers average weights")
        count += 1

    if any("lora_te3_text_model_encoder_layers" in key for key in state_dict.keys()):
        text_encoder_te1_layer_averages_and_max = (
            get_weight_vector_and_average_by_block(
                state_dict, ["lora_te3_text_model_encoder_layers"], [24]
            )
        )

        te3_x_name = []
        te3_avg = []

        print("\nText-Encoder TE3 layers average weights:")
        for layer, (avg, _) in text_encoder_te1_layer_averages_and_max.items():
            short_layer_name = layer.replace(
                "lora_te3_text_model_encoder_", "lora_te3_"
            )
            if isinstance(avg, str):
                print(f"{short_layer_name} average weight: {avg}")
            else:
                print(f"{short_layer_name} average weight: {avg:.16f}")
                te3_avg.append(round(avg, 16))
                te3_pattern = re.split(r"(.*?)_(\d+)$", short_layer_name)
                te3_x_name.append(te_plot_name[te3_pattern[1]] + te3_pattern[2])
        plt.subplot(row_counts, 1, count)
        plt.plot(te3_x_name, te3_avg, marker="o")
        plt.xticks(rotation=45)
        addlabels(te3_x_name, te3_avg)
        plt.title("Text-Encoder TE3 layers average weights")
        count += 1

    plt.tight_layout()
    plt.savefig(f"output/{filename.split('.')[0]}.png")
    img = Image.open(f"output/{filename.split('.')[0]}.png")
    img.show()


if __name__ == "__main__":
    main()
