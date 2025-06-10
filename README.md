# FreeTxt-Vi-KKM: A Toolkit for Vietnamese NLP

This repository contains a comprehensive toolkit for performing various Natural Language Processing (NLP) tasks on Vietnamese text, with some features also supporting English. The primary component is an interactive web application built with Streamlit. The project also includes notebooks for experimentation and development of NLP models.

## Key Features (Streamlit Application)

The interactive Streamlit application (`streamlit_app.py`) provides the following functionalities:

*   **Sentiment Analysis**: Classifies text as positive, negative, or neutral. Supports both Vietnamese and English.
*   **Text Summarization**: Generates abstractive summaries of long texts. Supports both Vietnamese and English.
*   **Word Cloud Generation**: Creates a word cloud from Vietnamese text, using `VnCoreNLP` for accurate word segmentation.
*   **Language Detection**: Automatically detects the language of the input text.
*   **Advanced Text Preprocessing**: Includes robust functions for cleaning and normalizing Vietnamese text, such as handling Unicode, standardizing tone marks, and removing irrelevant characters.

## Technology Stack

*   **Frontend**: Streamlit
*   **Core NLP/ML**: PyTorch, Transformers, `vncorenlp`
*   **Models**:
    *   Vietnamese Summarization: `VietAI/vit5-base-vietnews-summarization`
    *   Vietnamese Sentiment Analysis: `shenkha/FreeTxT-VisoBERT`
    *   English Sentiment Analysis: `nlptown/bert-base-multilingual-uncased-sentiment`
*   **Data Handling & Utilities**: Pandas, NLTK, Scikit-learn
*   **Visualization**: WordCloud, Matplotlib

---

## Setup and Installation

Follow these instructions to set up and run the Streamlit application on a local machine.

### 1. Prerequisites

*   **Python**: Version 3.10 is recommended.
*   **Git**: For cloning the repository.
*   **(Linux-based systems)**: You may need to install additional system libraries for word cloud generation. On Debian/Ubuntu, you can do this with:
    ```bash
    sudo apt-get update && sudo apt-get install -y libpangocairo-1.0-0
    ```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd FreeTxt-Vi-KKM
```

### 3. Set Up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install Python Dependencies

The required packages are listed in `streamlit_requirements.txt`. You can install them, along with other necessary packages not listed in the file, using pip.

```bash
pip install --upgrade pip
pip install streamlit torch transformers nltk pandas matplotlib wordcloud scikit-learn datasets evaluate langdetect summa vncorenlp==1.0.3
```

### 5. Download and Set Up VnCoreNLP

The Vietnamese word cloud feature depends on `VnCoreNLP`. The application expects a specific folder structure for these files.

**a. Create the necessary directories:**

From the root of the `FreeTxt-Vi-KKM` directory, run:

```bash
mkdir -p vncorenlp_files/models/wordsegmenter
```

**b. Download the required files:**

You need to download three files and place them in the structure you just created:

*   **VnCoreNLP-1.1.1.jar**: The main Java library.
    *   **Download Link**: [https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar](https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar)
    *   **Destination**: Place this file inside the `vncorenlp_files/` directory.

*   **vi-vocab**: The vocabulary model for the word segmenter.
    *   **Download Link**: [https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab](https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab)
    *   **Destination**: Place this file inside the `vncorenlp_files/models/wordsegmenter/` directory.

*   **wordsegmenter.rdr**: The main model for the word segmenter.
    *   **Download Link**: [https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr](https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr)
    *   **Destination**: Place this file inside the `vncorenlp_files/models/wordsegmenter/` directory.

After downloading, your directory structure should look like this:

```
FreeTxt-Vi-KKM/
|-- vncorenlp_files/
|   |-- VnCoreNLP-1.1.1.jar
|   |-- models/
|       |-- wordsegmenter/
|           |-- vi-vocab
|           |-- wordsegmenter.rdr
|-- streamlit_app.py
|-- ... (other files)
```

### 6. Download NLTK Data

The application uses the NLTK library for tokenization. The app attempts to download this automatically, but you can also do it manually by running this Python command:

```python
import nltk
nltk.download('punkt')
```

---

## How to Run the Application

Once you have completed the setup, you can launch the Streamlit application from your terminal:

```bash
streamlit run streamlit_app.py
```

Your web browser should open with the application interface.

## Experimentation Notebooks

This project contains several Jupyter notebooks used for development, evaluation, and experimentation:

*   `vncorenlp_test.ipynb`: A notebook for testing and experimenting with both `vncorenlp` and `py_vncorenlp` libraries for various NLP tasks.
*   `Preprocessing.ipynb`: Contains detailed steps and experiments for Vietnamese text preprocessing.
*   `trainSA.ipynb` / `finetuneSA.py`: Scripts and notebooks for training and fine-tuning the sentiment analysis models.
*   `evaluateSA.ipynb`: For evaluating the performance of the sentiment analysis models.
*   `finetuneSUM.py`: For fine-tuning the summarization models.

To run these, you will need to install Jupyter (`pip install jupyterlab`) and potentially other dependencies listed in `requirements.txt`. Note that the notebooks might have their own setup cells for installing packages and downloading models.

## Project Structure Overview

```
FreeTxt-Vi-KKM/
├── streamlit_app.py            # Main file for the Streamlit web application
├── streamlit_requirements.txt  # Python dependencies for the Streamlit app
├── vncorenlp_files/            # Directory for VnCoreNLP models (manual setup)
├── pyvncorenlp/                # Directory for py_vncorenlp models (automatic setup by library)
├── vncorenlp_test.ipynb        # Notebook for testing VnCoreNLP functionalities
├── Preprocessing.ipynb         # Notebook with text preprocessing experiments
├── finetuneSA.py               # Script for fine-tuning sentiment analysis models
├── finetuneSUM.py              # Script for fine-tuning summarization models
├── requirements.txt            # Dependencies for a separate Flask-based application
├── main.py                     # Entry point for the Flask application
└── ...                         # Other scripts, notebooks, and data files
```
