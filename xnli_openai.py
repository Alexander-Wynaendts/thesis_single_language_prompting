import pandas as pd
import openai
import unidecode
import time
import string
import re

sample_num = 200

start_time = time.time()

# OpenAI API configuration
openai.api_key = '****'

# Load the XNLI dataset
df = pd.read_csv('../database/xnli_dataset.csv')

# Function to generate entailment classification using OpenAI API with different prompts
def generate_entailment_openai_simple(language, premise, hypothesis):
    prompt = f"""
    Categorize the relationship between the following premise and hypothesis as "0" for Entailment, "1" for Neutral, and "2" for Contradiction.

    Premise: {premise}
    Hypothesis: {hypothesis}

    Provide only the numerical answer like this:
    Answer: 0 or 1 or 2
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    output = response['choices'][0]['message']['content'].strip()
    return output, prompt

def generate_entailment_openai_english_retell(language, premise, hypothesis):
    prompt = f"""
    I want you to act as an entailment classification expert for {language}.

    First, retell the premise in English. Then explore the following perspective:
    1. Analyse the reasons why the premise and the hyposthesis are entailing each other.
    2. Analyse the reasons why the premise and the hyposthesis are neutral to each other.
    3. Analyse the reasons why the premise and the hyposthesis are contradicting each other.
    4. Compare and contrast the 3 analysis and identify which one is the most accurate. Categorize the relationship between the premise and hypothesis as "0" for Entailment, "1" for Neutral, and "2" for Contradiction.

    Premise: {premise}
    Hypothesis: {hypothesis}

    Provide only the numerical answer like this:
    Answer: 0 or 1 or 2
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    output = response['choices'][0]['message']['content'].strip()
    return output, prompt

def generate_entailment_openai_english_translation(language, premise, hypothesis):
    translation_prompt = f"""
    Translate the following premise and hypothesis into English.

    Premise: {premise}
    Hypothesis: {hypothesis}

    Provide the translation without any other formatting or information in this format:

    Premise: the_premise
    Hypothesis: the_hypothesis
    """
    translation_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": translation_prompt},
        ],
    )
    translated_text = translation_response['choices'][0]['message']['content'].strip()

    premise_translated = re.search(r'Premise: (.*?)\n', translated_text).group(1)
    hypothesis_translated = re.search(r'Hypothesis: (.*?)$', translated_text).group(1)

    translated_prompt = f"""
    I want you to act as an entailment classification expert.

    Explore the following perspective:
    1. Analyse the reasons why the premise and the hyposthesis are entailing each other.
    2. Analyse the reasons why the premise and the hyposthesis are neutral to each other.
    3. Analyse the reasons why the premise and the hyposthesis are contradicting each other.
    4. Compare and contrast the 3 analysis and identify which one is the most accurate. Categorize the relationship between the premise and hypothesis as "0" for Entailment, "1" for Neutral, and "2" for Contradiction.

    Premise: {premise_translated}
    Hypothesis: {hypothesis_translated}

    Provide only the numerical answer like this:
    Answer: 0 or 1 or 2
    """

    entailment_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": translated_prompt},
        ],
    )
    output = entailment_response['choices'][0]['message']['content'].strip()
    return output, translated_prompt

def generate_entailment_openai_single_language(language, premise, hypothesis):
    template = """
    I want you to act as an entailment classification expert.

    Explore the following perspective:
    1. Analyse the reasons why the premise and the hyposthesis are entailing each other.
    2. Analyse the reasons why the premise and the hyposthesis are neutral to each other.
    3. Analyse the reasons why the premise and the hyposthesis are contradicting each other.
    4. Compare and contrast the 3 analysis and identify which one is the most accurate. Categorize the relationship between the premise and hypothesis as "0" for Entailment, "1" for Neutral, and "2" for Contradiction.

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
    translation_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": translation_prompt},
        ],
    )
    translated_template = translation_response['choices'][0]['message']['content'].strip()

    translated_prompt = translated_template.replace("{prem}", premise).replace("{hypoth}", hypothesis)

    entailment_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": translated_prompt},
        ],
    )
    output = entailment_response['choices'][0]['message']['content'].strip()
    return output, translated_prompt

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
    openai_entailment_simple, prompt_simple = generate_entailment_openai_simple(language, premise, hypothesis)
    openai_entailment_simple = clean_and_convert_label(openai_entailment_simple)
    openai_accuracy_simple = calculate_accuracy(label, openai_entailment_simple)
    results.append({
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'openai_entailment': openai_entailment_simple,
        'accuracy': openai_accuracy_simple,
        'prompt': prompt_simple,
        'prompt_method': 'simple'
    })

    # English prompt method
    openai_entailment_english, prompt_english = generate_entailment_openai_english_retell(language, premise, hypothesis)
    openai_entailment_english = clean_and_convert_label(openai_entailment_english)
    openai_accuracy_english = calculate_accuracy(label, openai_entailment_english)
    results.append({
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'openai_entailment': openai_entailment_english,
        'accuracy': openai_accuracy_english,
        'prompt': prompt_english,
        'prompt_method': 'english_retell'
    })

    # English translation prompt method
    openai_entailment_english_translation, prompt_english_translation = generate_entailment_openai_english_translation(language, premise, hypothesis)
    openai_entailment_english_translation = clean_and_convert_label(openai_entailment_english_translation)
    openai_accuracy_english_translation = calculate_accuracy(label, openai_entailment_english_translation)
    results.append({
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'openai_entailment': openai_entailment_english_translation,
        'accuracy': openai_accuracy_english_translation,
        'prompt': prompt_english_translation,
        'prompt_method': 'english_translation'
    })

    # Single language prompt method
    openai_entailment_single_language, prompt_single_language = generate_entailment_openai_single_language(language, premise, hypothesis)
    openai_entailment_single_language = clean_and_convert_label(openai_entailment_single_language)
    openai_accuracy_single_language = calculate_accuracy(label, openai_entailment_single_language)
    results.append({
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'openai_entailment': openai_entailment_single_language,
        'accuracy': openai_accuracy_single_language,
        'prompt': prompt_single_language,
        'prompt_method': 'single_language'
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to CSV
df_results.to_csv('../results/xnli_openai_results.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"XNLI OpenAI runtime is: {elapsed_time} seconds")
