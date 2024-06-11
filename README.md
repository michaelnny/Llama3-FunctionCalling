# Fine-tune Llama3 for Function Calling

This open-source project try to fine-tuning the llama3 model to support function calling.

# System Design Function Calling with LLMs

While function calling with large language models (LLMs) holds immense potential, crafting effective prompts and responses remains an art. Best practices are often guarded secrets. This guide `solution_design.md` dives into overcoming these challenges.

Here are the key design considerations:

- Function Metadata: How can we provide clear instructions about available functions to the LLM? (Solution: Inject serialized function metadata into the system prompt.)
- Initiating Function Calls: How can the LLM signal its intent to call a function? (Solution: LLM generates a JSON structure with a special "tool_uses" key.)
- Handling Results: How do we effectively communicate function call results back to the LLM? (Solution: Introduce a new "tool" role to add results after the function finishes.)

# Preparation

## Download and Prepare Meta weights

We need the original Llama3 instruct model weights. You can follow the guide on https://github.com/meta-llama/llama3

Once we have the weights ready, we then need to run the following script to convert the weight to native PyTorch, since we don't use fairscale in this project.

```bash
python3 scripts/convert_from_meta_weights.py --llama3-dir ~/models/Llama3/Meta-Llama-3-8B-Instruct --output-dir ~/models/Llama3/Llama-3-8B-Instruct
```

## Build Fine-tune Dataset

We use the glaive-function-calling-v2 dataset, but this dataset has limitations, including:

- Unstructured data: Conversations lack clear separation between roles (user vs. assistant), prompts, and responses.
- Invalid function calls: The dataset contains responses with incorrect or unfinished JSON structures for function calls.
- Unusable end-of-conversation: Some conversations conclude with user inputs, which are irrelevant for model training.
- Limited coverage: The dataset focuses on a single scenario (answering questions based on tool results) while we aim to cover broader function calling functionalities.

To address these issues, we developed a data cleaning script that:

- Removes invalid samples with malformed function call responses.
- Streamlines system prompt function metadata, following OpenAI's conventions for efficiency.
- Ensures a balanced dataset with assistant prompts for follow-up questions before function calls and assistant-initiated function calls.
- Standardizes conversation endings with assistant responses, facilitating clear prompt/response construction during training.

We can use the `clean_dataset.py` script to clean up and convert the dataset.

```bash
python3 scripts/clean_dataset.py --source-file ~/datasets/function_calling/glaive-function-calling-v2.json --save-file ./datasets/cleaned-glaive-function-calling-v2.json --remove-no-fc
```


Then, we can build the dataset using the `build_dataset.py` script.
```bash
python3 scripts/build_dataset.py --source-file ./datasets/cleaned-glaive-function-calling-v2.json --tokenizer-file ~/models/Llama3/Llama-3-8B-Instruct/tokenizer.model --save-dir ./datasets/glaive-function-calling-v2
```

**Tips**:

After you've build the dataset, you can run the `analyze_dataset.ipynb` inside `ideas` folder to check the dataset content.

# Start Fine-tuning with LoRA

## DeepSpeed with Multiple GPUs

We provide a simple script to fine-tune the llama3 model using DeepSpeed. By default, the script will load the `configs/ds_finetune.json` file. You should at least maintain the checkpoint files for both tokenizer and the model, and you may also change the batch size and other configurations.

Keep in mind that when using LoRA fine-tuning and DeepSpeed Zero optimizer, we can't run model validation if we use Stage3 zero, as it will break the LoRA specific weights.

```bash
# Fine-tune using DeepSpeed with multiple GPUS
DS_SKIP_CUDA_CHECK=1 deepspeed --num_gpus=1 ds_finetune.py
```

## Monitoring with Tensorboard

We can monitoring the training progress by using Tensorboard:

```bash
tensorboard --logdir ./logs
```

## Merge LoRA weights

After the training is finished, we need to merge the LoRA weights to the base model before we can use the model for inference.

### Convert DeepSpeed Zero checkpoints to a single checkpoint

This is required when using DeepSpeed, we might have multiple shard checkpoint states, and we need to merge them together.

We first need to merge the Zero checkpoints into a single checkpoint before we can continue to merge the LoRA weights.

```bash

mkdir ./checkpoints/converted_lora_state

cd ./checkpoints/ds_llama3_fc

python zero_to_fp32.py ./ ../converted_lora_state/lora_state_consolidated.pt


cp lora.json ../converted_lora_state/
cp params.json ../converted_lora_state/

```

Tips: by default, the script will looking for the latest checkpoint, if you want to do the conversion for other checkpoints, manually change the content of the `latest` file before you run the script.

### Merge LoRA weights

Then, we can use the following script to start the merge process.

```bash
python3 scripts/convert_lora_weights.py --config-dir ./checkpoints/converted_lora_state --base-ckpt-path ~/models/Llama3/Llama-3-8B-Instruct/consolidated.pth --lora-ckpt-path ./checkpoints/converted_lora_state/lora_state_consolidated.pt --output-dir ./checkpoints/finetuned_merged
```

## Single GPU

We also provide a simple script to fine-tune the llama3 model on a single GPU using LoRA and native PyTorch. By default, the script will load the `configs/finetune.json` file. You should at least maintain the checkpoint files for both tokenizer and the model.

```bash
# Fine-tune on a single GPU
python3 finetune.py
```

Notice we also need to merge the LoRA weights even if we only use a single GPU, but we don't need to run the script to convert DeepSpeed Zero checkpoints to a single checkpoint.

# Evaluation

We use a simple chat completion evaluation script adapted from the original llama3 project, where we made some changes to introduce function as tools.

```bash
# Using the fine-tuned model
python3 chat_completion.py --ckpt_dir ./checkpoints/finetuned_merged/ --tokenizer_path ~/models/Llama3/Llama-3-8B-Instruct/tokenizer.model


# Using the Meta model (weights converted using our script)
python3 chat_completion.py --ckpt_dir ~/models/Llama3/Llama-3-8B-Instruct  --tokenizer_path ~/models/Llama3/Llama-3-8B-Instruct/tokenizer.model
```

# Deployment

Once we are satisfied with the fine-tuned model, you might want to deploy it for inference. For example using Nvidia Triton server.

The preferred approach for deploying large language models (LLMs) on Triton is using the TensorRT-LLM backend. However, this requires checkpoints in either the native Meta Llama3 format or the Hugging Face (HF) format. Because right now TensorRT-LLM only provide options to convert these two model checkpoint formats into the special compute engine.

Since we're not using HF, we've developed a script, `convert_to_meta_weights.py` to convert custom fine-tuned model weights into a format compatible with TensorRT-LLM tools. This script essentially reverses the conversion done by `convert_from_meta_weights.py`.

```bash
python3 scripts/convert_to_meta_weights.py --config-file ./checkpoints/finetuned_merged/params.json --ckpt-file ./checkpoints/finetuned_merged/consolidated.pth --num-shards 1 --output-dir ~/models/Llama3/FineTuned-FC-Llama-3-8B-Instruct
```

# Limitations

- The dataset (glaive-function-calling-v2) only contains very simple function and use scenarios
- The dataset does not have sample where it involves calling multiple functions in a single response, or multiple different functions in a sequence
- The dataset does not have sample where the user provided some invalid data and LLM asking for corrections
- The dataset does not have sample where the LLM should correct itself if the tool call ended with error due to incorrect parameters
- We only tested on small model 8B version, and we didn't run sophisticated evaluations, like against other LLMs

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

For details about Llama3 model weights license, visit: https://github.com/meta-llama/llama3
