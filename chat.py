
import os
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          pipeline)


def get_pipeline(path, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True)

    print('Model loaded')

    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=[tokenizer.eos_token_id, 128001]
                         )
    return generator


SYS_PROMPT = None


def main(model_path: str,
         sys_prompt: str = SYS_PROMPT,
         max_new_tokens: int = 512,
         ** kwargs):

    print('model_path:', model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print('bos_token:', tokenizer.bos_token)
    print('bos_token_id:', tokenizer.bos_token_id)
    print('eos_token:', tokenizer.eos_token)
    print('eos_token_id:', tokenizer.eos_token_id)

    pipe = get_pipeline(model_path, tokenizer)

    messages = []
    if sys_prompt is not None:
        messages.append({"role": 'system', 'content': sys_prompt})

    while 1:
        input_ = input('\033[94mEnter instruction: ')
        if input_ == 'clear':
            messages = []
            if sys_prompt:
                messages.append({"role": 'system', 'content': sys_prompt})

            os.system('clear')
            continue
        elif input_ == 'exit':
            break
        messages.append({'role': 'user', 'content': input_})

        os.system('clear')
        for m in messages[:-1]:
            print('\033[92m' + m['role'] + ': ', m['content'])
        print('\033[93m' + 'User: ' + input_)
        start = time.time()
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        # print('template',text)
        result = pipe(text,
                      return_full_text=False,
                      clean_up_tokenization_spaces=True,
                      max_new_tokens=max_new_tokens,
                      do_sample=kwargs.get("do_sample", False),
                      **kwargs)[0]['generated_text']

        messages.append({'role': 'assistant', 'content': result})
        # print(messages)
        print('\033[95m' + 'Assistant: ' + result)
        print(f'elapsed time: {time.time() - start:.2f}s')


if __name__ == '__main__':
    import fire

    fire.Fire(main)
