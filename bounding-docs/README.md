<div align="center">

<h1>BoundingDocs</h1>

🔍 The largest spatially-annotated dataset for Document Question Answering

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.03403-b31b1b.svg)](https://arxiv.org/abs/2501.03403)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/letxbe/BoundingDocs)

</div>

## Dataset Description

BoundingDocs is a unified dataset for Document Question Answering (QA) that includes spatial annotations. It consolidates multiple public datasets from Document AI and Visually Rich Document Understanding (VRDU) domains. The dataset reformulates Information Extraction (IE) tasks into QA tasks, making it a valuable resource for training and evaluating Large Language Models (LLMs). Each question-answer pair is linked to its location in the document via bounding boxes, enhancing layout understanding and reducing hallucination risks in model outputs.

- **Curated by:** Simone Giovannini, Fabio Coppini, Andrea Gemelli, Simone Marinai
- **Language(s):** Primarily English, with multilingual support including Italian, Spanish, French, German, Portuguese, Chinese, and Japanese.
- **License:** CC-BY-4.0
- **Paper:** "BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations" by Giovannini et al.

The dataset has been curated during an internship of Simone Giovannini ([University of Florence](https://www.unifi.it/it)) at the company [Letxbe](https://letxbe.ai/).
<div align="center">
<img src="https://cdn.prod.website-files.com/655f447668b4ad1dd3d4b3d9/664cc272c3e176608bc14a4c_LOGO%20v0%20-%20LetXBebicolore.svg" alt="letxbe ai logo" width="200">
<img src="https://www.dinfo.unifi.it/upload/notizie/Logo_Dinfo_web%20(1).png" alt="Logo Unifi" width="200">
</div>

### 🌟 Highlights

- **Scale**: 48,151 documents, 237,437 pages, 249,016 QA pairs
- **Diversity**: 11 source datasets covering various document types
- **Spatial Awareness**: Precise bounding box annotations for all answers
- **Multilingual**: Support for 8 languages including English, Italian, Spanish, and more
- **Enhanced Questions**: AI-powered question rephrasing for linguistic diversity

### Direct Use

BoundingDocs is intended for tasks such as:
- Fine-tuning Document AI models for question answering with spatial context.
- Evaluating LLMs for visually rich document understanding.
- Studying the impact of spatial annotations on document comprehension tasks.
  

## 🔄 Version 2.0 Update

📢 **New Release: BoundingDocs v2.0**

We’ve released **version 2.0** of BoundingDocs with several important updates and improvements:

### ✅ What's New in v2.0

- **Rephrased Questions Standardized**  
  All datasets now include a `rephrased_question` field:
  - If the original question was already human-written (e.g., in DUDE, SP-DocVQA), the `rephrased_question` is identical to the `question`.
  - For non-English datasets like XFUND, rephrased versions were added using **Claude** to increase linguistic diversity.

- **Fixed MP-DocVQA Alignment**  
  In some `MP-DocVQA` entries, the order of `doc_images` and `doc_ocr` pages was mismatched during conversion.  
  This is now resolved — each image correctly aligns with its corresponding OCR result.

- **Accessing v2.0**  
  You can load the updated version using:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("letxbe/BoundingDocs", revision="v2.0")
  ```
  🧪 Note: The default version is still v1.0, used in the original paper. For reproducibility, use the default version unless you specifically need the updates in v2.0.

## 🚀 Quick Start

Load the dataset:
```python
from datasets import load_dataset
dataset = load_dataset("letxbe/boundingdocs")
sample = dataset['train'][0]
print(f"Document ID: {sample['doc_id']}")
```

Load and parse questions, rephrased questions and answers:
```python
# 'sample["Q&A"]' is a string that contains a JSON object. 
qa_data = json.loads(sample['Q&A'])
# After parsing, we can access the required fields from the JSON object.
print(f"Question: {qa_data[0]['question']}")  # Access the first question in the parsed JSON.
print(f"Rephrased Question: {qa_data[0]['rephrased_question']}")  # Access the rephrased version.
print(f"Answer Value: {qa_data[0]['answers'][0]['value']}")  # Access the value of the first answer.
print(f"Answer Location: {qa_data[0]['answers'][0]['location']}")  # Access the location of the first answers.
```

## Dataset Structure

### Data Fields

Each sample in BoundingDocs represents a whole document and contains the following fields:
 - **source**: The dataset where the document originates.
 - **doc_id**: The name of the file in its original dataset.
 - **doc_images**: A list of PIL images, one for each page in the document.
 - **doc_ocr**: Amazon Textract result of the document, in string format.
 - **Q&A**: The list of questions and answers described in JSON format.
Each Q&A pair includes:
 - **Questions**: The question posed to the model, in both template and rephrased forms.
 - **Answers**: A list of answers with associated bounding box coordinates normalized between 0 and 1000. The location bounding boxes format is `[width, height, x, y]` - where `(x,y)` is the bottom left corner.
 - **Page**: The page number where the answer is located.
An example looks as follows, with the exact JSON structure:
```json
{
  "question": "What is the Gross Amount?",
  "answers": [
    {
      "value": "$576,405.00",
      "location": [[90, 11, 364, 768]],  # [width, height, x, y]
      "page": 1
    }
  ],
  "rephrased_question": "What is the value of the Gross Amount?"
}
```
### 📊 Dataset Sources and Statistics
The dataset contains the following sources and statistics:
| Dataset            | Documents | Pages   | Questions  | Questions/Page | Questions/Document |
|--------------------|-----------|---------|------------|----------------|---------------------|
| Deepform          | 24,345    | 100,747 | 55,926     | 0.55           | 2.30                |
| DUDE              | 2,583     | 13,832  | 4,512      | 0.33           | 1.75                |
| FATURA            | 10,000    | 10,000  | 102,403    | 10.24          | 10.24               |
| FUNSD             | 199       | 199     | 1,542      | 7.75           | 7.75                |
| Kleister Charity  | 2,169     | 47,550  | 8,897      | 0.19           | 4.10                |
| Kleister NDA      | 337       | 2,126   | 696        | 0.33           | 2.07                |
| MP-DocVQA         | 5,203     | 57,643  | 31,597     | 0.55           | 6.07                |
| SP-DocVQA         | 266       | 266     | 419        | 1.58           | 1.58                |
| VRDU Ad Form      | 641       | 1,598   | 22,506     | 14.08          | 35.11               |
| VRDU Reg. Form    | 1,015     | 2,083   | 3,865      | 1.86           | 3.81                |
| XFUND             | 1,393     | 1,393   | 16,653     | 11.95          | 11.95               |
| **Total**         | **48,151**| **237,437** | **249,016** | **1.05**       | **5.17**            |

BoundingDocs is divided into training, validation, and test sets using an 80-10-10 split by document count, ensuring balanced layouts and question types across splits.

### ⚠️ Be aware of

While using the datasetm be aware that:
1. `doc_ocr` bounding box coordinates are normalized between 0 and 1 by Amazon Textract, while `answers` locations are between 0 and 1000!
2. In `DUDE`, `MP-DocVQA`, `SP-DocVQA` and `XFUND` sources you will find only `question` and not the rephrased ones!

More details in our paper!

## Dataset Creation

### Curation Rationale

BoundingDocs addresses the scarcity of extensive and diverse QA datasets in Document AI and the lack of precise spatial coordinates in existing datasets. <br>
By combining and standardizing data from multiple sources, BoundingDocs provides a consistent and enriched dataset for advanced document comprehension tasks.

### Data Collection and Processing

BoundingDocs integrates data from diverse datasets with various annotation formats. Processing steps include:
- Standardizing annotations into a unified format.
- Generating bounding box annotations using Amazon Textract.
- Rewriting questions with LLMs for linguistic diversity.

### Annotation Process

Bounding box annotations were generated through OCR (Amazon Textract), followed by alignment with existing annotations using Jaccard similarity. Questions were rephrased using Mistral 7B for enhanced linguistic variation.

### Personal and Sensitive Information

BoundingDocs includes documents from publicly available datasets.

## Bias, Risks, and Limitations

BoundingDocs may inherit biases from its source datasets. For example, certain fields may dominate specific datasets (e.g., financial terms in FATURA). 
Additionally, the dataset's multilingual support is limited, with the majority of questions in English.
Recommendations:

- Users should be aware of potential biases in question distributions and document types.
- When using BoundingDocs for multilingual tasks, consider the small proportion of non-English questions.

## Citation

If you use `BoundingDocs`, please cite:

```bibtex
@misc{giovannini2025boundingdocsunifieddatasetdocument,
      title={BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations}, 
      author={Simone Giovannini and Fabio Coppini and Andrea Gemelli and Simone Marinai},
      year={2025},
      eprint={2501.03403},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.03403}, 
}
```
