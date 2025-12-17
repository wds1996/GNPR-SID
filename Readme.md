# Repository Structure: V1 and V2 Releases

This repository provides **two independent versions**:  
**V1** corresponds to the implementation used in our KDD 2025 paper *“Generative Next POI Recommendation with Semantic ID.”*  
**V2** is a subsequently improved version that incorporates further refinements and modular updates.

---

## 🥇 V1 — Original Release

Directory: **`V1/`**

This version contains the **initial implementation**, including:

- RQ-VAE quantization
- Sample NYC dataset
- Basic instructions for LLM fine-tuning

---

## 🚀 V2 — Updated Release

Directory: **`V2/`**

This version provides the **latest and recommended implementation**, featuring:

- Reorganized project structure
- POI embedding module under
- SID module with projection-based cosine quantization and EMA smoothing
- Sample LLM fine-tuning code
- Support for SID-LLM semantic alignment or direct full fine-tuning

---


### **Cite Us**

```bibtex
@inproceedings{wang2025generative,
  title={Generative Next POI Recommendation with Semantic ID},
  author={Wang, Dongsheng and Huang, Yuxi and Gao, Shen and Wang, Yifan and Huang, Chengrui and Shang, Shuo},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={2904--2914},
  year={2025}
}
```

