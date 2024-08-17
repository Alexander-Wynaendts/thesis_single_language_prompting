import pandas as pd
import google.generativeai as genai
import unidecode
import time
import string
import re

sample_num = 10

start_time = time.time()

# Configure Gemini API key
gemini_api_key = '****'
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel('gemini-pro')

# Load the combined dataset
df = pd.read_csv('../database/xcopa_dataset.csv')

# Function to generate cause-effect identification using Gemini API with different prompts
def generate_cause_effect_gemini_simple(language, premise, choice1, choice2, question):
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        Knowing this premise "{premise}", what is the {question}?

        A: {choice1}
        B: {choice2}

        Answer with the integer "0" for "A" and with the integer "1" for "B".

        Provide only the numerical answer like this:
        Answer: 0 or 1
        """
        response = chat.send_message(prompt)
        output = response.text.strip()
        return output, prompt
    except Exception as e:
        print(f"Error")
        return "", prompt

def generate_cause_effect_gemini_english_retell(language, premise, choice1, choice2, question):
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        I want you to act as a cause-effect identification expert for {language}.

        First, retell the premise in English. Then explore the following perspective:
        1. Analyse the reasons why A might be the {question} of the premise
        2. Analyse the reasons why B might be the {question} of the premise
        3. Compare and contrast the 2 analysis and identify what is the correct {question}, A or B.

        Premise: {premise}
        A: {choice1}
        B: {choice2}

        Answer with the integer "0" for "A" and with the integer "1" for "B".

        Provide only the numerical answer like this:
        Answer: 0 or 1
        """
        response = chat.send_message(prompt)
        output = response.text.strip()
        return output, prompt
    except Exception as e:
        print(f"Error")
        return "", prompt

def generate_cause_effect_gemini_single_language(language, premise, choice1, choice2, question):
    try:
        chat = model.start_chat(history=[])
        # Template to be translated
        template = """
        I want you to act as a cause-effect identification expert.

        First, retell the premise in English. Then explore the following perspective:
        1. Analyse the reasons why A might be the cause/effect of the premise
        2. Analyse the reasons why B might be the cause/effect of the premise
        3. Compare and contrast the 2 analysis and identify what is the correct cause/effect, A or B.

        Premise: {prem}
        A: {cho1}
        B: {cho2}

        Answer with the integer "0" for "A" and with the integer "1" for "B".

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

        # Insert the premise, question, choices into the translated template
        translated_prompt = translated_template.replace("{prem}", premise).replace("{cho1}", choice1).replace("{cho2}", choice2)

        # Use the translated prompt to generate the cause-effect identification
        cause_effect_response = chat.send_message(translated_prompt)
        output = cause_effect_response.text.strip()
        return output, translated_prompt
    except Exception as e:
        print(f"Error")
        return "", translation_prompt

def generate_cause_effect_gemini_english_translation(language, premise, choice1, choice2, question):
    try:
        chat = model.start_chat(history=[])
        translation_prompt = f"""
        Translate the following premise and choices into English.

        Premise: {premise}
        A: {choice1}
        B: {choice2}

        Provide the translation without any other formatting or information in this format:

        Premise: the_premise
        A: choice1
        B: choice2
        """
        translation_response = chat.send_message(translation_prompt)
        translated_text = translation_response.text.strip()

        premise_translated = re.search(r'Premise: (.*?)\n', translated_text).group(1)
        choice1_translated = re.search(r'A: (.*?)\n', translated_text).group(1)
        choice2_translated = re.search(r'B: (.*?)$', translated_text).group(1)

        translated_prompt = f"""
        I want you to act as a cause-effect identification expert.

        Explore the following perspective:
        1. Analyse the reasons why A might be the cause/effect of the premise
        2. Analyse the reasons why B might be the cause/effect of the premise
        3. Compare and contrast the 2 analysis and identify what is the correct cause/effect, A or B.

        Premise: {premise_translated}
        A: {choice1_translated}
        B: {choice2_translated}

        Answer with the integer "0" for "A" and with the integer "1" for "B".

        Provide only the numerical answer like this:
        Answer: 0 or 1
        """

        cause_effect_response = chat.send_message(translated_prompt)
        output = cause_effect_response.text.strip()
        return output, translated_prompt
    except Exception as e:
        print(f"Error")
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

# Generate cause-effect identifications and evaluate using accuracy
results = []

for index, row in df_sampled.iterrows():
    print(f"Processing index: {index}")

    premise = row['premise']
    choice1 = row['choice1']
    choice2 = row['choice2']
    question = row['question']
    label = row['label']
    language = row['language']

    # Simple prompt method
    gemini_cause_effect_simple, prompt_simple = generate_cause_effect_gemini_simple(language, premise, choice1, choice2, question)
    gemini_cause_effect_simple = clean_and_convert_label(gemini_cause_effect_simple)
    gemini_accuracy_simple = calculate_accuracy(label, gemini_cause_effect_simple)
    results.append({
        'language': language,
        'premise': premise,
        'choice1': choice1,
        'choice2': choice2,
        'question': question,
        'label': label,
        'gemini_cause_effect': gemini_cause_effect_simple,
        'accuracy': gemini_accuracy_simple,
        'prompt': prompt_simple,
        'prompt_method': 'simple'
    })

    # English prompt method
    gemini_cause_effect_english, prompt_english = generate_cause_effect_gemini_english_retell(language, premise, choice1, choice2, question)
    gemini_cause_effect_english = clean_and_convert_label(gemini_cause_effect_english)
    gemini_accuracy_english = calculate_accuracy(label, gemini_cause_effect_english)
    results.append({
        'language': language,
        'premise': premise,
        'choice1': choice1,
        'choice2': choice2,
        'question': question,
        'label': label,
        'gemini_cause_effect': gemini_cause_effect_english,
        'accuracy': gemini_accuracy_english,
        'prompt': prompt_english,
        'prompt_method': 'english_retell'
    })

    # English translation prompt method
    gemini_cause_effect_english_translation, prompt_english_translation = generate_cause_effect_gemini_english_translation(language, premise, choice1, choice2, question)
    gemini_cause_effect_english_translation = clean_and_convert_label(gemini_cause_effect_english_translation)
    gemini_accuracy_english_translation = calculate_accuracy(label, gemini_cause_effect_english_translation)
    results.append({
        'language': language,
        'premise': premise,
        'choice1': choice1,
        'choice2': choice2,
        'question': question,
        'label': label,
        'gemini_cause_effect': gemini_cause_effect_english_translation,
        'accuracy': gemini_accuracy_english_translation,
        'prompt': prompt_english_translation,
        'prompt_method': 'english_translation'
    })

    # Single language prompt method
    gemini_cause_effect_single_language, prompt_single_language = generate_cause_effect_gemini_single_language(language, premise, choice1, choice2, question)
    gemini_cause_effect_single_language = clean_and_convert_label(gemini_cause_effect_single_language)
    gemini_accuracy_single_language = calculate_accuracy(label, gemini_cause_effect_single_language)
    results.append({
        'language': language,
        'premise': premise,
        'choice1': choice1,
        'choice2': choice2,
        'question': question,
        'label': label,
        'gemini_cause_effect': gemini_cause_effect_single_language,
        'accuracy': gemini_accuracy_single_language,
        'prompt': prompt_single_language,
        'prompt_method': 'single_language'
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to CSV
df_results.to_csv('../results/xcopa_gemini_results.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"XCOPA Gemini runtime is: {elapsed_time} seconds")
