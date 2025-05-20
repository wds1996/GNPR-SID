The full codebase is still under organization. Currently, the following components have been made available:

- **RQ-VAE Module**  
  The core implementation of the Residual Vector Quantized Variational Autoencoder (RQ-VAE) is provided.  
  To train a custom codebook, use the script [`code/train_rqvae.py`](./code/train_rqvae.py).  
  After training, the mapping from discrete token IDs to semantic IDs can be generated using [`codebook.py`](./codebook.py).

- **Sample Dataset**  
  A sample dataset based on **NYC** data is included for demonstration and evaluation purposes.

- **Data Preprocessing**  
  The preprocessing pipeline is provided in the following Jupyter notebooks:  
  - [`dataprocess.ipynb`](./path/to/dataprocess.ipynb): for raw data cleaning and formatting  
  - [`data2json.ipynb`](./path/to/data2json.ipynb): for converting processed data into model-ready JSON format

- **Model Fine-tuning**  
  We adopt the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework for fine-tuning large language models.  
  You can reproduce the fine-tuning and evaluation processes by using our dataset along with the official instructions provided in the LLaMA-Factory repository.

