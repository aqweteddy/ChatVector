import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


def add_chat_vector(
    base: PreTrainedModel,
    chat_vector_path: str,
    ratio: float,
    skip_embed: bool = False,
    special_tokens_map: dict[int, int] = None
):
    chat_vector = torch.load(f'{chat_vector_path}/pytorch_model.bin')
    
    print(special_tokens_map)

    for n, p in base.named_parameters():
        if 'embed_tokens' in n or 'word_embeddings' in n:
            if not skip_embed:
                assert p.data.shape == chat_vector['chat_vector'][
                    n].shape, "embeds_token shape mismatch. Use --skip_embed to skip embedding layers."
                p.data += ratio * chat_vector['chat_vector'][n]
            elif special_tokens_map:
                for k, v in special_tokens_map.items():
                    p.data[k] += ratio * \
                        chat_vector['chat_embed'][v]
        elif 'lm_head' in n:
            if not skip_embed:
                p.data += ratio * chat_vector['chat_vector'][n]
            elif special_tokens_map:
                for k, v in special_tokens_map.items():
                    p.data[k] += ratio * \
                        chat_vector['chat_lmhead'][v]
        else:
            p.data += ratio * chat_vector['chat_vector'][n]

    return base, chat_vector['cfg']


def main(
    base_model_path: str,
    chat_vector_path: list[str],
    output_path: str,
    ratio: list[float] = [1],
    skip_embed: bool = False,
    special_tokens_map: dict[int, int] = None
):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype='auto')

    if special_tokens_map:
        for k, v in special_tokens_map.items():
            base_model.get_input_embeddings().weight.data[k] = torch.zeros(
                base_model.config.hidden_size
            )
            base_model.get_output_embeddings().weight.data[k] = torch.zeros(
                base_model.config.hidden_size
            )

    for cv_path, r in zip(chat_vector_path, ratio):
        base_model, cfg = add_chat_vector(
            base_model, cv_path, r, skip_embed, special_tokens_map)

    # set tokenizer
    # last chat_vector_path as chat_template.
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    chat_tokenizer = AutoTokenizer.from_pretrained(cfg['chat_model_path'])

    if chat_tokenizer.chat_template is None:
        logger.warning('chat_tokenizer.chat_template is None')
    else:
        logger.info(f'chat_template: {tokenizer.chat_template}')
        tokenizer.chat_template = chat_tokenizer.chat_template
        tokenizer.eos_token = chat_tokenizer.eos_token
        tokenizer.eos_token_id = chat_tokenizer.eos_token_id

    base_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size='8GB'
    )

    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
