# Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages

<p align="center">
    <img src="https://img.shields.io/badge/Code_License-MIT-blue">
</p>

This repository contains the official code for the ACL 2024 paper: [Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages](https://arxiv.org/pdf/2310.04799).

## Requirements

* torch
* transformers
* fire

To install the required packages, run the following commands:

```bash
CUDA=cu118 # change to your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/$CUDA

# If you do not need to use `chat.py`, you can install the no-cuda torch version.
pip install torch

pip install transformers fire
```

## Usage

### Extracting the Chat Vector

To extract the chat vector, use the following command:

```bash
BASE_MODEL_PATH=meta-llama/Meta-Llama-3-8B
CHAT_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
CHAT_VECTOR_PATH=ckpt_tv/llama3-8b-instruct

python extract_chat_vector.py $BASE_MODEL_PATH $CHAT_MODEL_PATH $CHAT_VECTOR_PATH
```

### Adding the Chat Vector

To add the chat vector to the model, use the following command:

```bash
CP_MODEL_PATH=ckpt/llama3-8b_cp
OUTPUT_PATH=ckpt/llama3-8b-cp_cv-llama3

python add_chat_vector.py $CP_MODEL_PATH "['$CHAT_VECTOR_PATH']" $OUTPUT_PATH \
--ratio "[1]"  # chat vector ratio
```

If you encounter issues with outputting the target language, please lower the `ratio` setting.

### Skip Embedding

In cases where you need to continue pretraining with extended word embeddings, you can use the `--skip_embed` option to avoid adding the embedding and `lm_head` layer:

```bash
CP_MODEL_PATH=ckpt/llama3-8b_cp
OUTPUT_PATH=ckpt/llama3-8b-cp_cv-llama3

python add_chat_vector.py $CP_MODEL_PATH "['$CHAT_VECTOR_PATH']" $OUTPUT_PATH --skip_embed True
```

If certain special tokens in the chat template (such as `<|eot_id|>`) are not trained during continual pretraining, you should set `special_tokens_map` to replace the CP model's special tokens embedding with the chat model's tokens. For example, with `llama3`:

```bash
python add_chat_vector.py $CP_MODEL_PATH "['$CHAT_VECTOR_PATH']" $OUTPUT_PATH \
--ratio "[1]" \  # chat vector ratio
--skip_embed True \
--special_tokens_map "{128006:128006,128007:128007,128009:128009}"  # {'CP_MODEL_TOKEN_ID':'CHAT_MODEL_TOKEN_ID'}
```

If the model does not generate text properly, consider fine-tuning the model with the added chat vector.

### Merging Multiple Chat Vectors

To merge multiple chat vectors, use the following command:

```bash
OUTPUT_PATH=ckpt/llama3-8b-cp_cv-llama3-openhermess
CV1_PATH=ckpt_tv/llama3-8b-instruct
CV2_PATH=ckpt_tv/llama3-8b-openhermess

python add_chat_vector.py $CP_MODEL_PATH "['$CV1_PATH','$CV2_PATH']" $OUTPUT_PATH \
--ratio "[0.5,0.5]"
# Enable `--skip_embed` and `--special_tokens_map` if needed
```

* set `chat_template` to `$CV2_PATH`'s chat template.

## Chat Script

```bash
python chat.py \ 
$OUTPUT_PATH \  # model path
# --sys_prompt "你是一個樂於助人的助理。" \  # system prompt
# --<other generation config>
```

## Citation

If you find this paper helpful, please use the following citation:

```bibtex
@misc{huang2024chat,
      title={Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages}, 
      author={Shih-Cheng Huang* and Pin-Zu Li* and Yu-Chi Hsu and Kuang-Ming Chen and Yu Tung Lin and Shih-Kai Hsiao and Richard Tzong-Han Tsai and Hung-yi Lee*},
      year={2024},
      eprint={2310.04799},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement

We appreciate the support and resources provided by the TAIDE project.
