import pandas as pd
import google.generativeai as genai
import unidecode
import time
import string
import re

sample_num = 200

start_time = time.time()

# Configure Gemini API key
gemini_api_key = '****'
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel('gemini-pro')

# Load the XNLI dataset
df = pd.read_csv('../database/xnli_dataset.csv')

# Function to generate entailment classification using Gemini API with different prompts
def generate_entailment_gemini_simple(language, premise, hypothesis):
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        Categorize the relationship between the following premise and hypothesis as "0" for Entailment, "1" for Neutral, and "2" for Contradiction.

        Premise: {premise}
        Hypothesis: {hypothesis}

        Provide only the numerical answer like this:
        Answer: number
        """
        response = chat.send_message(prompt)
        output = response.text.strip()
        return output, prompt
    except Exception as e:
        print(f"Error")
        return "", prompt

def generate_entailment_gemini_english_retell(language, premise, hypothesis):
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        I want you to act as an entailment classification expert for {language}.

        First, retell the premise and hypothesis in English. Then explore the following perspectives:
        1. Analyse the reasons why the premise and the hypothesis are entailing each other.
        2. Analyse the reasons why the premise and the hypothesis are neutral to each other.
        3. Analyse the reasons why the premise and the hypothesis are contradicting each other.
        4. Compare and contrast the 3 analyses and identify which one is the most accurate. Categorize the relationship between the premise and hypothesis as "0" for Entailment, "1" for Neutral, and "2" for Contradiction.

        Premise: {premise}
        Hypothesis: {hypothesis}

        Provide only the numerical answer like this:
        Answer: 0 or 1 or 2
        """
        response = chat.send_message(prompt)
        output = response.text.strip()
        return output, prompt
    except Exception as e:
        print(f"Error")
        return "", prompt

def generate_entailment_gemini_english_translation(language, premise, hypothesis):
    try:
        chat = model.start_chat(history=[])
        translation_prompt = f"""
        Translate the following premise and hypothesis into English.

        Premise: {premise}
        Hypothesis: {hypothesis}

        Provide the translation without any other formatting or information in this format:

        Premise: the_premise
        Hypothesis: the_hypothesis
        """
        translation_response = chat.send_message(translation_prompt)
        translated_text = translation_response.text.strip()

        premise_translated = re.search(r'Premise: (.*?)\n', translated_text).group(1)
        hypothesis_translated = re.search(r'Hypothesis: (.*?)$', translated_text).group(1)

        translated_prompt = f"""
        I want you to act as an entailment classification expert.

        Explore the following perspectives:
        1. Analyse the reasons why the premise and the hypothesis are entailing each other.
        2. Analyse the reasons why the premise and the hypothesis are neutral to each other.
        3. Analyse the reasons why the premise and the hypothesis are contradicting each other.
        4. Compare and contrast the 3 analyses and identify which one is the most accurate. Categorize the relationship between the premise and hypothesis as "0" for Entailment, "1" for Neutral, and "2" for Contradiction.

        Premise: {premise_translated}
        Hypothesis: {hypothesis_translated}

        Provide only the numerical answer like this:
        Answer: 0 or 1 or 2
        """

        entailment_response = chat.send_message(translated_prompt)
        output = entailment_response.text.strip()
        return output, translated_prompt
    except Exception as e:
        print(f"Error")
        return "", translated_prompt

def generate_entailment_gemini_single_language(language, premise, hypothesis):
    try:
        chat = model.start_chat(history=[])
        template = """
        I want you to act as an entailment classification expert.

        Explore the following perspectives:
        1. Analyse the reasons why the premise and the hypothesis are entailing each other.
        2. Analyse the reasons why the premise and the hypothesis are neutral to each other.
        3. Analyse the reasons why the premise and the hypothesis are contradicting each other.
        4. Compare and contrast the 3 analyses and identify which one is the most accurate. Categorize the relationship between the premise and hypothesis as "0" for Entailment, "1" for Neutral, and "2" for Contradiction.

        Premise: {prem}
        Hypothesis: {hypoth}

        Provide only the numerical answer like this:
        Answer: 0 or 1 or 2
        """

        translation_prompt = f"""
        Translate the following template into {language}, but do not translate the placeholders within curly braces.

        Provide the translation without any other formatting or information, just the translated template:

        "{template}"
        """
        translation_response = chat.send_message(translation_prompt)
        translated_template = translation_response.text.strip()

        translated_prompt = translated_template.replace("{prem}", premise).replace("{hypoth}", hypothesis)

        entailment_response = chat.send_message(translated_prompt)
        output = entailment_response.text.strip()
        return output, translated_prompt
    except Exception as e:
        print(f"Error")
        return "", translated_prompt

# Function to preprocess text
def preprocess_text(text):
    text = text.lower().strip()
    text = unidecode.unidecode(text)
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    text = text.replace('-', ' ')
    text = ' '.join(text.split())
    return text

# Function to calculate accuracy
def calculate_accuracy(true_label, generated_label):
    return 1 if true_label == generated_label else 0

# Function to clean and convert the generated label
def clean_and_convert_label(label):
    label = re.sub(r'[^0-9]', '', label)
    return int(label) if label.isdigit() else -1

# Initialize an empty DataFrame to hold the sampled data
df_sampled = pd.DataFrame()

# List of languages in the dataset
languages = df['language'].unique()

# Sample 1 row per language
for lang in languages:
    df_lang = df[df['language'] == lang]
    df_lang_sampled = df_lang.sample(n=sample_num)
    df_sampled = pd.concat([df_sampled, df_lang_sampled], ignore_index=True)

# Generate entailment classifications and evaluate using accuracy
results = []

for index, row in df_sampled.iterrows():
    print(f"Processing index: {index}")

    premise = row['premise']
    hypothesis = row['hypothesis']
    label = row['label']
    language = row['language']

    # Simple prompt method
    gemini_entailment_simple, prompt_simple = generate_entailment_gemini_simple(language, premise, hypothesis)
    gemini_entailment_simple = clean_and_convert_label(gemini_entailment_simple)
    gemini_accuracy_simple = calculate_accuracy(label, gemini_entailment_simple)
    results.append({
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'gemini_entailment': gemini_entailment_simple,
        'accuracy': gemini_accuracy_simple,
        'prompt': prompt_simple,
        'prompt_method': 'simple'
    })

    # English prompt method
    gemini_entailment_english, prompt_english = generate_entailment_gemini_english_retell(language, premise, hypothesis)
    gemini_entailment_english = clean_and_convert_label(gemini_entailment_english)
    gemini_accuracy_english = calculate_accuracy(label, gemini_entailment_english)
    results.append({
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'gemini_entailment': gemini_entailment_english,
        'accuracy': gemini_accuracy_english,
        'prompt': prompt_english,
        'prompt_method': 'english_retell'
    })

    # English translation prompt method
    gemini_entailment_english_translation, prompt_english_translation = generate_entailment_gemini_english_translation(language, premise, hypothesis)
    gemini_entailment_english_translation = clean_and_convert_label(gemini_entailment_english_translation)
    gemini_accuracy_english_translation = calculate_accuracy(label, gemini_entailment_english_translation)
    results.append({
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'gemini_entailment': gemini_entailment_english_translation,
        'accuracy': gemini_accuracy_english_translation,
        'prompt': prompt_english_translation,
        'prompt_method': 'english_translation'
    })

    # Single language prompt method
    gemini_entailment_single_language, prompt_single_language = generate_entailment_gemini_single_language(language, premise, hypothesis)
    gemini_entailment_single_language = clean_and_convert_label(gemini_entailment_single_language)
    gemini_accuracy_single_language = calculate_accuracy(label, gemini_entailment_single_language)
    results.append({
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'gemini_entailment': gemini_entailment_single_language,
        'accuracy': gemini_accuracy_single_language,
        'prompt': prompt_single_language,
        'prompt_method': 'single_language'
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to CSV
df_results.to_csv('../results/xnli_gemini_results.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"XNLI Gemini runtime is: {elapsed_time} seconds")
