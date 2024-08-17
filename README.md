# Prompt Engineering with Multilingual Support in LLM Models

## Project Overview
This project investigates the efficacy of prompt engineering techniques in Large Language Models (LLMs) when the prompts are partially or fully translated into less commonly used languages. The analysis is conducted using OpenAI's ChatGPT 4 and Google's Gemini 1.5 models across four distinct tasks: MSGM, XCOPA, XNLI, and PAWSX.

## Files and Directories

- `main.py`: Entry point for the project, orchestrating the entire experiment workflow.
- `data_analysis.py`: Contains functions for analyzing the results of the experiments.
- `mgsm_openai.py`, `xcopa_openai.py`, `xnli_openai.py`, `pawsx_openai.py`: Scripts for running the respective tasks using OpenAI's ChatGPT 4.
- `mgsm_gemini.py`, `xcopa_gemini.py`, `xnli_gemini.py`, `pawsx_gemini.py`: Scripts for running the respective tasks using Google's Gemini 1.5.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Running the Experiments:**
    - For each task specify the amount of samples needed in the variable `sample_num`.
    - To run a specific task with OpenAI's model, execute the corresponding script. For example:
        ```bash
        python mgsm_openai.py
        ```
    - To run a specific task with Google's model, execute the corresponding script. For example:
        ```bash
        python mgsm_gemini.py
        ```
    - To run a all tasks with Google's and OpenAI's model, execute the following script:
    ```bash
    python main.py
    ```

2. **Analyzing Results:**
    - Use the `data_analysis.py` script to process and analyze the results from the experiments:
        ```bash
        python data_analysis.py
        ```

## Project Structure

├── data_analysis.py
├── main.py
├── mgsm_openai.py
├── xcopa_openai.py
├── xnli_openai.py
├── pawsx_openai.py
├── mgsm_gemini.py
├── xcopa_gemini.py
├── xnli_gemini.py
├── pawsx_gemini.py
└── requirements.txt
