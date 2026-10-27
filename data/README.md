# Dataset (not included)

This project uses the **Online Retail II** dataset from the UCI Machine Learning Repository (dataset ID 502), licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**.

- Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Citation: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- The workbook is approximately 43.5 MB. For local runs, place `online_retail_II.xlsx` or a supported CSV directly in this folder.
- The loader accepts `.xlsx` (both sheets are read and de-duplicated) or `.csv`.

For the Render deployment blueprint, `scripts/build_demo_service.py` fetches the source dataset through the UCI dataset API during the build, writes a temporary CSV, builds the model artifacts, and deletes the temporary CSV after the pipeline completes. The downloaded source dataset and generated artifact files are not committed to Git.
