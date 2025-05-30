# EMGLLM: Data-to-Text Alignment for Electromyogram Diagnosis Generation with Medical Numerical Data Encoding


## ETM Dataset

ETM (Electromyogram Table Mart) is a high quality EMG diagnostic report dataset derived from Huashan Hospital Affiliated to Fudan University with high authenticity, accuracy, and authority, which contains a total of 17,250 diagnostic reports from 2006 to 2013, and each data includes:

$\bullet$ Basic information of real anonymized patients (age, gender, and height).

$\bullet$ EMG tables (EMG and NCV tests) from the real EMG examination in the hospital.

$\bullet$ Diagnosis (\emph{Findings} and \emph{Impression}) personally written by experienced physicians.

The full dataset is further proportionally divided into training, validation, and testing set, with data volumes of 13800, 1725, and 1725 respectively, which can effectively support medical data-to-text research.

ETM-17k: https://huggingface.co/datasets/fangbingzi/ETM-17k

For the training of language models, you can use ETM-17k-LLM version: https://huggingface.co/datasets/fangbingzi/ETM-17k-LLM
