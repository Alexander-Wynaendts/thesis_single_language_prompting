import pandas as pd
import os

# List of datasets and models
datasets = ['xnli', 'pawsx', 'xcopa', 'mgsm']
models = ['openai', 'gemini']

# Directories
input_dir = '../results'
output_dir = '..'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to generate the combined reports for both models
def generate_combined_reports(df1, df2, dataset, output_dir):
    # Combine the dataframes for both models
    combined_df = pd.concat([df1, df2])

    # Reorder the columns to put 'model' first
    cols = combined_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('model')))
    combined_df = combined_df[cols]

    # 1. Average accuracy by language
    mean_accuracy_by_language = combined_df.groupby(['model', 'language'])['accuracy'].mean().reset_index()
    language_report_file = os.path.join(output_dir, f'{dataset}_language_report.csv')
    mean_accuracy_by_language.to_csv(language_report_file, index=False)

    # 2. Average accuracy by prompt method
    mean_accuracy_by_prompt_method = combined_df.groupby(['model', 'prompt_method'])['accuracy'].mean().reset_index()
    prompt_report_file = os.path.join(output_dir, f'{dataset}_prompt_report.csv')
    mean_accuracy_by_prompt_method.to_csv(prompt_report_file, index=False)

    # 3. Average accuracy by the combination of language and prompt method
    mean_accuracy_by_language_and_prompt = combined_df.groupby(['model', 'language', 'prompt_method'])['accuracy'].mean().reset_index()
    combined_report_file = os.path.join(output_dir, f'{dataset}_language_prompt_combination_report.csv')
    mean_accuracy_by_language_and_prompt.to_csv(combined_report_file, index=False)

# Process each dataset to create combined reports for both models
for dataset in datasets:
    model1 = models[0]
    model2 = models[1]

    # Construct the filenames
    file_path_model1 = os.path.join(input_dir, f'{dataset}_{model1}_results.csv')
    file_path_model2 = os.path.join(input_dir, f'{dataset}_{model2}_results.csv')

    # Check if the files exist
    if os.path.exists(file_path_model1) and os.path.exists(file_path_model2):
        # Load the CSV files
        df1 = pd.read_csv(file_path_model1)
        df1['model'] = model1
        df2 = pd.read_csv(file_path_model2)
        df2['model'] = model2

        # Generate the combined reports
        generate_combined_reports(df1, df2, dataset, output_dir)
