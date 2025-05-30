# EMGLLM: Data-to-Text Alignment for Electromyogram Diagnosis Generation with Medical Numerical Data Encoding

Electromyography (EMG) tables play a crucial role in diagnosing muscle and nerve disorders. However, the complex, continuous numerical data in EMG reports pose significant challenges for current Large Language Models (LLMs), which struggle to interpret such structured medical information effectively.

**EMGLLM is a specialized data-to-text model designed to bridge this gap.** It introduces an EMG Alignment Encoder, which simulates how medical professionals compare patient test values with reference ranges, aligning the data into embeddings that reflect a patient's health status more accurately.

To support research and development in this area, we also introduce ETM, a large-scale dataset containing 17,250 real-world EMG cases paired with diagnostic reports. Our experiments show that EMGLLM significantly outperforms baseline models on EMG table interpretation and diagnostic text generation, offering a new paradigm for automatic diagnosis from structured medical data.

## ETM Dataset

ETM-17k: https://huggingface.co/datasets/fangbingzi/ETM-17k

For the training of language models, you can use ETM-17k-LLM version: https://huggingface.co/datasets/fangbingzi/ETM-17k-LLM

<img src="/figure/data_example.jpg" alt="An EMG diagnostic report example in ETM" style="width:50%;" />


ETM (Electromyogram Table Mart) is a high quality EMG diagnostic report dataset derived from Huashan Hospital Affiliated to Fudan University with high authenticity, accuracy, and authority, which contains a total of 17,250 diagnostic reports from 2006 to 2013, and each data includes:

$\bullet$ Basic information of real anonymized patients (age, gender, and height).

$\bullet$ EMG tables (EMG and NCV tests) from the real EMG examination in the hospital.

$\bullet$ Diagnosis ***Findings*** and ***Impression*** personally written by experienced physicians.

The full dataset is further proportionally divided into training, validation, and testing set, with data volumes of 13800, 1725, and 1725 respectively, which can effectively support medical data-to-text research.


