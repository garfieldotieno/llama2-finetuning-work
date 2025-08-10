# 🚀 LLaMA 2 Fine-Tuning on Google Colab

This repository provides a complete workflow for fine-tuning **Meta’s LLaMA 2** models using **LoRA/QLoRA** on **Google Colab** with a T4 GPU.  
It covers everything from getting model access to deploying your trained model.

---

## 📚 Contents
1. [LLaMA 2 Access](#1-llama-2-access)
2. [Dataset](#2-dataset)
3. [Google Colab Setup](#3-google-colab-setup)
4. [Training](#4-training)
5. [Merging & Exporting](#5-merging--exporting)
6. [Deploying](#6-deploying)
7. [License](#7-license)
8. [Acknowledgement](#8-acknowledgement)

---

## 📜 1. LLaMA 2 Access

Before you can download or fine-tune LLaMA 2, you need to request and be granted access.

### Step 1 — Accept Meta’s LLaMA 2 License
- Visit: [https://ai.meta.com/llama](https://ai.meta.com/llama)
- Fill out the request form and agree to the license terms.
- Wait for Meta’s approval email.

### Step 2 — Request Access on Hugging Face
- Go to the [LLaMA 2 Hugging Face page](https://huggingface.co/meta-llama).
- Click **Access request** and accept the additional terms.

### Step 3 — Login in Google Colab
After installing `huggingface_hub`, authenticate your Hugging Face account:

```python
from huggingface_hub import login
login()  # Enter your Hugging Face token when prompted
```

---

## 📂 2. Dataset

### Default Dataset — OpenAssistant Guanaco
- **Size:** ~1 GB download  
- **Format:** Instruction–response pairs  
- Hugging Face dataset link: [OpenAssistant Guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

```python
from datasets import load_dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")
```

---

## 🖥️ 3. Google Colab Setup

### Step 1 — Open the Colab Notebook
- Open your prepared Google Colab notebook for LLaMA 2 fine-tuning.

### Step 2 — Enable GPU
- Go to: **Runtime → Change runtime type → GPU**
- **Recommended GPU:** T4 for cost and performance balance.

### Step 3 — (Optional) Mount Google Drive
This allows you to save model checkpoints and outputs directly to your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🏋️ 4. Training

Run the fine-tuning script with your chosen parameters.  
Below is an example using **LLaMA 2 7B** with the **OpenAssistant Guanaco** dataset and **QLoRA**:

```bash
python finetune.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name timdettmers/openassistant-guanaco \
  --use_4bit True \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --output_dir ./results
```

---

## 🔗 5. Merging & Exporting

After training, you can merge the **LoRA** weights into the base model to create a standalone model for inference or deployment.

```python
from peft import AutoPeftModelForCausalLM

# Load the trained model with LoRA adapters
model = AutoPeftModelForCausalLM.from_pretrained("./results/final_checkpoints")

# Merge LoRA weights into the base model
model = model.merge_and_unload()

# Save the merged model
model.save_pretrained(
    "./results/final_merged_checkpoint",
    safe_serialization=True
)
```

---

## 🚀 6. Deploying

You can deploy your fine-tuned LLaMA 2 model by uploading it to the **Hugging Face Hub** for easy sharing and inference.

### Step 1 — Create a New Model Repository
- Go to [https://huggingface.co/new](https://huggingface.co/new)  
- Create a **Model** repo and copy its clone URL.

### Step 2 — Push Your Model

```bash
# Log in to Hugging Face
huggingface-cli login

# Enable Git LFS for large files
git lfs install

# Clone your new model repository
git clone https://huggingface.co/your-username/your-model

# Copy merged model files into the repo
cp -r ./results/final_merged_checkpoint/* your-model/

# Commit and push
cd your-model
git add .
git commit -m "Add fine-tuned LLaMA 2 model"
git push
```

---

## 📄 7. License

- **Code:** Licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).  
- **Model Weights:** Usage requires approval under [Meta’s LLaMA 2 license](https://ai.meta.com/llama/license/).  
- **Dataset:** Ensure compliance with the dataset’s original license terms before use or redistribution.

---

---

## 🙏 8. Acknowledgement

- **Meta AI** — for developing and releasing [LLaMA 2](https://ai.meta.com/llama).  
- **Hugging Face** — for [Transformers](https://huggingface.co/docs/transformers/index), [Datasets](https://huggingface.co/docs/datasets/index), and the [Hugging Face Hub](https://huggingface.co/).  
- **Tim Dettmers** — for the [Guanaco dataset](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) and the [QLoRA method](https://arxiv.org/abs/2305.14314).

---