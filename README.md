Okay, here's a draft for your `README.md` file summarizing the project's progress and next steps.

---

# PaperGen AI - Automated IEEE Formatter & AI Reviewer

This project aims to create a tool that automatically formats academic papers into the IEEE conference style and provides AI-powered feedback to improve the paper's quality.

## Project Goal

To build a web application where users can input their pre-written paper content (text, figures, references) and receive:
1.  A perfectly formatted PDF document adhering to the strict `IEEEtran` standard.
2.  Optional, actionable suggestions for improving the paper's structure and clarity, based on an AI model trained to predict paper acceptance.

**Note:** This tool *formats* and *reviews*; it does **not** write or generate content for the user.

## Current Progress

### 1. AI Reviewer Engine (Proof of Concept)
* **Data Sourcing & Labeling:** Manually downloaded and labeled a small dataset (~60 papers) consisting of accepted papers (from IEEE Xplore conference proceedings) and rejected proxies (recent arXiv pre-prints not found in proceedings).
* **Preprocessing:** Implemented PDF text extraction using `PyMuPDF`.
* **Embedding:** Converted full paper text into numerical vectors (embeddings) using the `BAAI/bge-m3` model via `sentence-transformers`, employing a chunking and averaging strategy. Embeddings stored in `.npy` format, labels in `.csv`.
* **Model Training:** Trained an `XGBoost` binary classifier on the embeddings to predict paper status (Accepted/Rejected).
* **Performance:** Achieved **~96% accuracy** on the initial test split, demonstrating the feasibility of learning patterns related to paper acceptance. The trained model is saved as `ai_reviewer_model_tuned.joblib`.
* **Technology Stack:** Python, `PyMuPDF`, `sentence-transformers`, `numpy`, `pandas`, `xgboost`, `scikit-learn`.

### 2. Automated Formatting Engine
* **Environment Setup:** Configured a local LaTeX environment using MiKTeX.
* **Core Logic:** Developed a Python script (`build_paper.py`) that takes dictionary-based text content as input.
* **LaTeX Assembly:** The script programmatically inserts the user's text into a standard `IEEEtran` LaTeX template (`.tex` file) and creates a corresponding bibliography file (`.bib`).
* **Automated Compilation:** Utilizes Python's `subprocess` module to automatically run `pdflatex` and `bibtex` commands, generating the final formatted PDF from the assembled `.tex` project.
* **Technology Stack:** Python, MiKTeX (`pdflatex`, `bibtex`), `subprocess`.

---

## Next Steps / To Work On

1.  **Improve AI Model Robustness:**
    * **Problem:** The current model shows high accuracy on the small test set but misclassified a known real-world rejected paper (potentially overfitting or lacking diverse examples).
    * **Action:** Add more diverse data (especially confirmed rejected papers), potentially augment existing data, and perform more rigorous hyperparameter tuning (beyond the initial `GridSearchCV`) focusing on regularization parameters (`gamma`, `reg_alpha`, `reg_lambda`). Re-evaluate performance on a wider range of real-world examples.

2.  **Develop Frontend Interface:**
    * **Problem:** No user interface currently exists.
    * **Action:** Build an HTML/CSS/JavaScript frontend with text areas for each paper section, file uploads for figures, and controls to trigger the formatting and review process.

3.  **Build Backend API (Flask):**
    * **Problem:** The formatting and AI model logic exist only as standalone scripts.
    * **Action:** Wrap the `assemble_latex_project`, `compile_latex_to_pdf`, and model prediction logic into API endpoints using a web framework like Flask. This API will handle requests from the frontend.

4.  **Enhance Embedding Strategy:**
    * **Problem:** Chunking and averaging might lose fine-grained details.
    * **Action:** Research and potentially implement alternative embedding techniques like weighted averaging (prioritizing abstract/conclusion) or hierarchical embeddings if the current approach proves insufficient after model retraining. Explore newer or more specialized embedding models if available.

---
