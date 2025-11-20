```markdown
# BLIP-2 Fine-Tuned for Image Captioning 🖼️

This repository contains code and notebooks to **fine-tune the BLIP-2** (Bootstrapped Language-Image Pretraining) model for image captioning on the **Flickr8k dataset**, using **PEFT (LoRA)** to enable efficient training.

---

## 🚀 Overview

- **Base model**: BLIP-2 from Salesforce (vision encoder + Q-Former + frozen LLM)  
- **Task**: Image captioning  
- **Dataset**: Flickr8k (8,000 images, each with 5 captions) :contentReference[oaicite:0]{index=0}  
- **Fine-tuning method**: LoRA (Parameter-Efficient Fine-Tuning) :contentReference[oaicite:1]{index=1}  
- **Frameworks / Libraries**:  
  - `transformers` (Hugging Face)  
  - `datasets` for data loading :contentReference[oaicite:2]{index=2}  
  - `peft` for LoRA :contentReference[oaicite:3]{index=3}  
  - `bitsandbytes` (optional, for memory-efficient training) :contentReference[oaicite:4]{index=4}  

---

## 📂 Repository Structure

```

BLIP-2-Fine-Tuned/
│
├── data/                  # Scripts or instructions to load / preprocess Flickr8k
│
├── notebooks/             # Jupyter notebooks for training & inference
│   └── fine_tune.ipynb     # Notebook to fine-tune BLIP-2 using LoRA
│
├── src/
│   ├── train.py            # Script to train / fine-tune the model
│   ├── inference.py        # Script or module for inference
│   ├── dataset.py          # Dataset classes and data loading utilities
│   └── utils.py            # Helper functions (tokenizer, image transforms, etc.)
│
├── requirements.txt        # Python dependencies
│
└── README.md               # This file

````

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Muavia1/BLIP-2-Fine-Tuned.git
   cd BLIP-2-Fine-Tuned
````

2. Create a Python environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Fine-Tuning / Training

To fine-tune the model:

```bash
python src/train.py \
  --dataset_dir /path/to/flickr8k/ \
  --output_dir ./models/blip2-finetuned/ \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1
```

**Parameters Explanation:**

* `dataset_dir`: Directory where Flickr8k images + captions are stored
* `output_dir`: Where to save the fine-tuned model
* `lora_rank`, `lora_alpha`, `lora_dropout`: LoRA hyperparameters

---

## 🔍 Inference

After fine-tuning, you can generate captions given an image:

```bash
python src/inference.py \
  --image_path /path/to/image.jpg \
  --model_dir ./models/blip2-finetuned/ \
  --max_length 30
```

The script will load the fine-tuned model and output a generated caption for the provided image.

---

## 📓 Notebook Usage

* Open `notebooks/fine_tune.ipynb`
* Follow the steps to load data, define the model, apply LoRA, train, and run inference
* Useful for quick experiments / visualizations

---

## 🧪 Evaluation & Results

* After training, you can evaluate generated captions using standard metrics like **BLEU**, **ROUGE**, or **CIDEr**.
* Use the inference script to generate predictions and compare them with ground-truth captions from Flickr8k.

---

## ✅ Why Use This

* **Efficient Training**: LoRA-based fine-tuning means you don’t have to update the full model, saving memory and compute.
* **Practical Application**: Fine-tuned for image captioning — useful in accessibility, image search, and content description.
* **Reproducible**: Notebook + scripts make it easy to reproduce results or extend to other datasets.
* **Modular**: You can easily adapt the code for other vision-language tasks (e.g., VQA) or datasets.

---

## 🚀 Future Work / Extensions

* Fine-tune on **larger datasets** (e.g., Flickr30k, MS-COCO)
* Experiment with **other PEFT methods** (e.g., QLoRA)
* Add support for **visual question answering (VQA)** or **image-text retrieval**
* Deploy as a **public inference API** or **Streamlit / Gradio app**

---

## 🤝 Contributing

Contributions are very welcome!
Feel free to:

* Open issues (e.g., bugs, feature requests)
* Submit pull requests
* Share improved hyperparameters, training tricks, or evaluation scripts

---

## 📚 References

* BLIP-2: Bootstrapped Language-Image Pretraining ([Learning Muse by Mehdi Seyfi][1])
* PEFT / LoRA method for efficient fine-tuning ([Medium][2])
* Flickr8k dataset for image captioning ([Medium][2])

---

## 📝 License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute freely.

---


Do you want me to add them?

[1]: https://mseyfi.github.io/posts/VLM/BLIP2.html?utm_source=chatgpt.com "Learning Muse by Mehdi Seyfi"
[2]: https://medium.com/%40muaviaijaz8/fine-tuning-blip-2-for-image-captioning-with-the-flickr8k-dataset-f4e4906a67d2?utm_source=chatgpt.com "Fine-Tuning BLIP-2 for Image Captioning with the Flickr8k Dataset | by Muavia Abdul Moiz | Medium"
