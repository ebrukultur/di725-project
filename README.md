# Vision–Language Model Improvements for Remote Sensing Image Captioning  
**DI 725 Project:** Transformers and Attention-Based Deep Networks

**Author:** Ebru Kültür Başaran  
**Affiliation:** Middle East Technical University, Informatics Institute  


---

## 📖 Project Overview  
This repository contains the deliverables for the DI725 final project, which aims to enhance the PaliGemma vision–language model for remote sensing image captioning (RSIC) using the RISC dataset.  


## Repository Structure

The project is organized as follows:

```text
di725-project/
├── code/                        # python scripts for training and evaluation
│   └── train_lora.py            # fine-tuning paligemma with lora
├── data/                        # (not committed) placeholder for RISC dataset
├── notebooks/                   # Jupyter notebooks
│   └── eda_risc.ipynb           # EDA on caption lengths, vocab, image-caption pairs
├── reports/                     # Report files
│   └── DI725_project_p1_1741420_report.pdf        # 2‑page IEEE PDF of Phase 1 deliverables
├── figures/                     # Generated figures
│   └── caption_length_histogram.png
├── requirements.txt             # Python dependencies
└── README.md                    # This file
