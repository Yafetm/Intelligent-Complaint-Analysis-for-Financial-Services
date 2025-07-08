# Intelligent Complaint Analysis for Financial Services

## Overview
This project develops a Retrieval-Augmented Generation (RAG) chatbot for CrediTrust Financial to analyze customer complaints from the CFPB dataset across five products: Credit Card, Personal Loan, Buy Now Pay Later, Savings Account, and Money Transfers. Due to hardware constraints, a sampled dataset was used.

## Directory Structure
- `data/`: Contains `sampled_complaints.csv` and `filtered_complaints.csv`.
- `src/`: Scripts for tasks.
  - `task1_eda_preprocessing.py`: EDA and preprocessing.
  - `task2_chunking_embedding.py`: Text chunking and embedding.
  - `task3_rag_pipeline.py`: RAG pipeline with `distilgpt2`.
  - `task4_chatbot_interface.py`: Streamlit interface.
  - `sample_complaints.py`: Sampling script.
- `vector_store/`: FAISS index and metadata.
- `narrative_length_distribution.png`: Visualization.
- `rag_test_outputs.txt`: Task 3 outputs.
- `interim_report.tex`, `interim_report.pdf`: Interim report.
- `final_report.tex`, `final_report.pdf`: Final report.

## Setup
1. Activate virtual environment: `venv\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run scripts: `python src/task1_eda_preprocessing.py`, etc.
4. Run chatbot: `streamlit run src/task4_chatbot_interface.py`

## Tasks
- **Task 1**: Sampled 400,000 complaints with narratives, filtered to ~196,298 rows for Credit Card, Personal Loan, Savings Account, and Money Transfers. Buy Now Pay Later complaints were not sampled, likely due to missing narratives.
- **Task 2**: Generated ~600,000 chunks from filtered complaints, embedded with `all-MiniLM-L6-v2`, stored in FAISS.
- **Task 3**: Implemented RAG pipeline with `distilgpt2`, improved summary coherence with prompt engineering.
- **Task 4**: Delivered a stable Streamlit interface for querying complaints and viewing summaries.

## Results
The chatbot retrieves and summarizes complaints for Credit Card, Savings Account, Money Transfers, and Personal Loan. Buy Now Pay Later retrieval is not supported due to sampling limitations.

## Limitations
- **Buy Now Pay Later**: Missing in sampled dataset, possibly due to absent narratives in the original dataset or sampling issues.
- **Model**: `distilgpt2` produces basic summaries; stronger models could improve coherence.
- **Dataset**: Sampled 400,000 rows due to hardware constraints (5GB full dataset).

## Next Steps
- Process full 5GB dataset with more powerful hardware.
- Include Buy Now Pay Later by adjusting sampling or verifying narrative presence.

## GitHub Repository
[https://github.com/Yafetm/Intelligent-Complaint-Analysis-for-Financial-Services](https://github.com/Yafetm/Intelligent-Complaint-Analysis-for-Financial-Services)