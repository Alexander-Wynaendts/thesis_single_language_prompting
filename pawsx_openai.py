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

# Load the combined dataset
df = pd.read_csv('../database/pawsx_dataset.csv')

# Function to generate paraphrase identification using Gemini API with different prompts
def generate_paraphrase_gemini_simple(language, sentence1, sentence2):
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        Are these two sentences paraphrases of each other? Answer "1" for yes and "0" for no.

        Sentence 1: {sentence1}
        Sentence 2: {sentence2}

        Provide only the numerical answer like this:
        Answer: 0 or 1
        """
        response = chat.send_message(prompt)
        output = response.text.strip()
        return output, prompt
    except Exception as e:
        print(f"""
              Error on the prompt:
              {prompt}
              """)
        return "", prompt

def generate_paraphrase_gemini_english_retell(language, sentence1, sentence2):
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        I want you to act as a paraphrase identification expert for {language}.

        First, retell the sentences in English. Then, determine if these two sentences are paraphrases of each other. Answer "1" for yes and "0" for no.

        First, retell the sentences in English. Then, explore the following perspective:
        1. Analyse the reasons why the two sentences are paraphrases of each other.
        2. Analyse the reasons why the two sentences are not paraphrases of each other.
        3. Compare and contrast the 2 analysis and identify which one is the most accurate. Answer "1" if it is a paraphrase and "0" if it is not a paraphrase.

        Sentence 1: {sentence1}
        Sentence 2: {sentence2}

        Provide only the numerical answer like this:
        Answer: 0 or 1
        """
        response = chat.send_message(prompt)
        output = response.text.strip()
        return output, prompt
    except Exception as e:
        print(f"""
              Error on the prompt:
              {prompt}
              """)
        return "", prompt

def generate_paraphrase_gemini_single_language(language, sentence1, sentence2):
    try:
        chat = model.start_chat(history=[])
        # Template to be translated
        template = """
        I want you to act as a paraphrase identification expert.

        Explore the following perspective:
        1. Analyse the reasons why the two sentences are paraphrases of each other.
        2. Analyse the reasons why the two sentences are not paraphrases of each other.
        3. Compare and contrast the 2 analysis and identify which one is the most accurate. Answer "1" if it is a paraphrase and "0" if it is not a paraphrase.

        Sentence 1: {sent1}
        Sentence 2: {sent2}

        Provide only the numerical answer like this:
        Answer: 0 or 1
        """

        # Translate the template
        translation_prompt = f"""
        Translate the following template into {language}, but do not translate the placeholders within curly braces.

        Provide the translation without any other formatting or information, just the translated template:

        "{template}"
        """
        translation_response = chat.send_message(translation_prompt)
        translated_template = translation_response.text.strip()

        # Insert the sentences into the translated template
        translated_prompt = translated_template.replace("{sent1}", sentence1).replace("{sent2}", sentence2)

        # Use the translated prompt to generate the paraphrase identification
        paraphrase_response = chat.send_message(translated_prompt)
        output = paraphrase_response.text.strip()
        return output, translated_prompt
    except Exception as e:
        print(f"""
              Error on the prompt:
              {translation_prompt}
              """)
        return "", translation_prompt

def generate_paraphrase_gemini_english_translation(language, sentence1, sentence2):
    try:
        chat = model.start_chat(history=[])
        translation_prompt = f"""
        Translate the following sentences into English.

        Sentence 1: {sentence1}
        Sentence 2: {sentence2}

        Provide the translation without any other formatting or information in this format:

        Sentence 1: translated_sentence1
        Sentence 2: translated_sentence2
        """
        translation_response = chat.send_message(translation_prompt)
        translated_text = translation_response.text.strip()

        sentence1_translated = re.search(r'Sentence 1: (.*?)\n', translated_text).group(1)
        sentence2_translated = re.search(r'Sentence 2: (.*?)$', translated_text).group(1)

        translated_prompt = f"""
        I want you to act as a paraphrase identification expert.

        Explore the following perspective:
        1. Analyse the reasons why the two sentences are paraphrases of each other.
        2. Analyse the reasons why the two sentences are not paraphrases of each other.
        3. Compare and contrast the 2 analysis and identify which one is the most accurate. Answer "1" if it is a paraphrase and "0" if it is not a paraphrase.

        Sentence 1: {sentence1_translated}
        Sentence 2: {sentence2_translated}

        Provide only the numerical answer like this:
        Answer: 0 or 1
        """

        paraphrase_response = chat.send_message(translated_prompt)
        output = paraphrase_response.text.strip()
        return output, translated_prompt
    except Exception as e:
        print(f"""
              Error on the prompt:
              {translation_prompt}
              """)
        return "", translation_prompt

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
    label = re.sub(r'[^0-9]', '', label)  # Remove any non-numeric characters
    return int(label) if label.isdigit() else -1  # Convert to int if possible, else return -1 for invalid

# Initialize an empty DataFrame to hold the sampled data
df_sampled = pd.DataFrame()

# List of languages in the dataset
languages = df['language'].unique()

# Sample 1 row per language
for lang in languages:
    df_lang = df[df['language'] == lang]
    df_lang_sampled = df_lang.sample(n=sample_num)
    df_sampled = pd.concat([df_sampled, df_lang_sampled], ignore_index=True)

# Generate paraphrase identifications and evaluate using accuracy
results = []

for index, row in df_sampled.iterrows():
    print(f"Processing index: {index}")

    sentence1 = row['sentence1']
    sentence2 = row['sentence2']
    label = row['label']
    language = row['language']

    # Simple prompt method
    gemini_paraphrase_simple, prompt_simple = generate_paraphrase_gemini_simple(language, sentence1, sentence2)
    gemini_paraphrase_simple = clean_and_convert_label(gemini_paraphrase_simple)
    gemini_accuracy_simple = calculate_accuracy(label, gemini_paraphrase_simple)
    results.append({
        'language': language,
        'sentence1': sentence1,
        'sentence2': sentence2,
        'label': label,
        'gemini_paraphrase': gemini_paraphrase_simple,
        'accuracy': gemini_accuracy_simple,
        'prompt': prompt_simple,
        'prompt_method': 'simple'
    })

    # English prompt method
    gemini_paraphrase_english, prompt_english = generate_paraphrase_gemini_english_retell(language, sentence1, sentence2)
    gemini_paraphrase_english = clean_and_convert_label(gemini_paraphrase_english)
    gemini_accuracy_english = calculate_accuracy(label, gemini_paraphrase_english)
    results.append({
        'language': language,
        'sentence1': sentence1,
        'sentence2': sentence2,
        'label': label,
        'gemini_paraphrase': gemini_paraphrase_english,
        'accuracy': gemini_accuracy_english,
        'prompt': prompt_english,
        'prompt_method': 'english_retell'
    })

    # English translation prompt method
    gemini_paraphrase_english_translation, prompt_english_translation = generate_paraphrase_gemini_english_translation(language, sentence1, sentence2)
    gemini_paraphrase_english_translation = clean_and_convert_label(gemini_paraphrase_english_translation)
    gemini_accuracy_english_translation = calculate_accuracy(label, gemini_paraphrase_english_translation)
    results.append({
        'language': language,
        'sentence1': sentence1,
        'sentence2': sentence2,
        'label': label,
        'gemini_paraphrase': gemini_paraphrase_english_translation,
        'accuracy': gemini_accuracy_english_translation,
        'prompt': prompt_english_translation,
        'prompt_method': 'english_translation'
    })

    # Single language prompt method
    gemini_paraphrase_single_language, prompt_single_language = generate_paraphrase_gemini_single_language(language, sentence1, sentence2)
    gemini_paraphrase_single_language = clean_and_convert_label(gemini_paraphrase_single_language)
    gemini_accuracy_single_language = calculate_accuracy(label, gemini_paraphrase_single_language)
    results.append({
        'language': language,
        'sentence1': sentence1,
        'sentence2': sentence2,
        'label': label,
        'gemini_paraphrase': gemini_paraphrase_single_language,
        'accuracy': gemini_accuracy_single_language,
        'prompt': prompt_single_language,
        'prompt_method': 'single_language'
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to CSV
df_results.to_csv('../results/pawsx_gemini_results.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"PAWS-X OpenAI runtime is: {elapsed_time} seconds")
