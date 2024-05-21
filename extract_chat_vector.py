import torch, os
from transformers import  AutoModelForCausalLM


def extract(
    base_model_path: str,
    chat_model_path: str,
    output_path: str,
):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype='auto')
    ft_model = AutoModelForCausalLM.from_pretrained(chat_model_path, torch_dtype='auto')
    
    chat_vector_params = {
        'chat_embed': base_model.get_input_embeddings().weight,
        'chat_lmhead': base_model.get_output_embeddings().weight,
        'chat_vector': {},
        'cfg': {
            'base_model_path': base_model_path,
            'chat_model_path': chat_model_path,
        }
    }
    
    for (n1, p1), (n2, p2) in zip(base_model.named_parameters(),ft_model.named_parameters()):
        chat_vector_params['chat_vector'][n1] = p2.data - p1.data
    
    os.makedirs(output_path, exist_ok=True)
    torch.save(
        chat_vector_params,
        f"{output_path}/pytorch_model.bin",
    )


if __name__ == '__main__':
    from fire import Fire
    Fire(extract)