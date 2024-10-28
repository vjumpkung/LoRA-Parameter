import torch
from safetensors.torch import load_file
import argparse
import time

start_time = time.time()

def count_parameters(state_dict, search_terms):
    relevant_params = {k: v for k, v in state_dict.items() if any(term in k for term in search_terms)}
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

def main():
    parser = argparse.ArgumentParser(description="Calculate parameters of LoRA components.")
    parser.add_argument("-i", "--input", required=True, help="Path to the .safetensors file.")
    args = parser.parse_args()

    lora_model_path = args.input
    state_dict = load_file(lora_model_path)

    unet_params, _ = count_parameters(state_dict, ['lora_unet'])
    conv_params, _ = count_parameters(state_dict, ['conv'])
    unet_single_params, _ = count_parameters(state_dict, ['lora_unet_single'])
    unet_double_params, _ = count_parameters(state_dict, ['lora_unet_double'])
    text_encoder_1_params, _ = count_parameters(state_dict, ['lora_te_text_model_encoder', 'lora_te1_text_model_encoder'])
    text_encoder_2_params, _ = count_parameters(state_dict, ['lora_te2'])
    text_encoder_3_params, _ = count_parameters(state_dict, ['lora_te3'])

    print(f"UNet                    : {format_parameters(unet_params)}")
    print(f"Conv layer UNet         : {format_parameters(conv_params)}")
    print(f"UNet single block [Flux]: {format_parameters(unet_single_params)}")
    print(f"UNet double block [Flux]: {format_parameters(unet_double_params)}")
    print(f"Text-Encoder 1 Clip_L   : {format_parameters(text_encoder_1_params)}")
    print(f"Text-Encoder 2 Clip_G   : {format_parameters(text_encoder_2_params)}")
    print(f"Text-Encoder 3 T5       : {format_parameters(text_encoder_3_params)}")
    
    end_time = time.time()
    latency = end_time - start_time
    print
    print(f"Latency: {latency:.2f} seconds")

if __name__ == "__main__":
    main()
