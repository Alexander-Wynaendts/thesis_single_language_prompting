import pandas as pd
import google.generativeai as genai
import time
import re

sample_num = 200

start_time = time.time()

# Configure Gemini API key
gemini_api_key = '****'
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel('gemini-pro')

# Load the mgsm dataset
df = pd.read_csv('../database/mgsm_dataset.csv')

# Function to generate answers using Gemini API
def generate_answer_gemini_simple(question):
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        {question}
        """
        response = chat.send_message(prompt)
        output = response.text.strip()
        return output, prompt
    except Exception as e:
        print(f"Error")
        return "", prompt

def generate_answer_gemini_english_retell(language, question):
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        I want you to act as a mathematical problem-solving expert.

        First, retell the question in English. Then, solve the question by thinking step by step.

        Question: {question}
        """
        response = chat.send_message(prompt)
        output = response.text.strip()
        return output, prompt
    except Exception as e:
        print(f"Error")
        return "", prompt

def generate_answer_gemini_single_language(language, question):
    try:
        chat = model.start_chat(history=[])
        # Template to be translated
        template = """
        I want you to act as a mathematical problem-solving expert.

        Solve the question by thinking step by step.

        Question: {quest}
        """

        # Translate the template
        translation_prompt = f"""
        Translate the following template into {language}, but do not translate the placeholders within curly braces.

        Provide the translation without any other formatting or information, just the translated template:

        "{template}"
        """
        translation_response = chat.send_message(translation_prompt)
        translated_template = translation_response.text.strip()

        # Insert the question into the translated template
        translated_prompt = translated_template.replace("{quest}", question)

        # Use the translated prompt to generate the answer
        answer_response = chat.send_message(translated_prompt)
        output = answer_response.text.strip()
        return output, translated_prompt
    except Exception as e:
        print(f"Error")
        return "", template

def generate_answer_gemini_english_translation(language, question):
    try:
        chat = model.start_chat(history=[])
        translation_prompt = f"""
        Translate the following question into English.

        Question: {question}

        Provide the translation without any other formatting or information in this format:

        Question: translated_question
        """
        translation_response = chat.send_message(translation_prompt)
        translated_text = translation_response.text.strip()

        question_translated = re.search(r'Question: (.*?)$', translated_text).group(1)

        translated_prompt = f"""
        I want you to act as a mathematical problem-solving expert.

        Solve the question by thinking step by step.

        Question: {question_translated}
        """

        answer_response = chat.send_message(translated_prompt)
        output = answer_response.text.strip()
        return output, translated_prompt
    except Exception as e:
        print(f"Error")
        return "", translation_prompt

# Initialize an empty DataFrame to hold the sampled data
df_sampled = pd.DataFrame()

# List of languages in the dataset
languages = df['language'].unique()

# Sample 1 row per language
for lang in languages:
    df_lang = df[df['language'] == lang]
    df_lang_sampled = df_lang.sample(n=sample_num)
    df_sampled = pd.concat([df_sampled, df_lang_sampled], ignore_index=True)

# Generate answers and store in results
results = []

for index, row in df_sampled.iterrows():
    print(f"Processing index: {index}")

    question = row['question']
    answer_number = row['answer_number']
    language = row['language']

    # Simple prompt method
    gemini_answer_simple, prompt_simple = generate_answer_gemini_simple(question)
    results.append({
        'language': language,
        'question': question,
        'answer_number': answer_number,
        'gemini_answer': gemini_answer_simple,
        'prompt': prompt_simple,
        'prompt_method': 'simple'
    })

    # English prompt method
    gemini_answer_english, prompt_english = generate_answer_gemini_english_retell(language, question)
    results.append({
        'language': language,
        'question': question,
        'answer_number': answer_number,
        'gemini_answer': gemini_answer_english,
        'prompt': prompt_english,
        'prompt_method': 'english'
    })

    # English translation prompt method
    gemini_answer_english_translation, prompt_english_translation = generate_answer_gemini_english_translation(language, question)
    results.append({
        'language': language,
        'question': question,
        'answer_number': answer_number,
        'gemini_answer': gemini_answer_english_translation,
        'prompt': prompt_english_translation,
        'prompt_method': 'english_translation'
    })

    # Single language prompt method
    gemini_answer_single_language, prompt_single_language = generate_answer_gemini_single_language(language, question)
    results.append({
        'language': language,
        'question': question,
        'answer_number': answer_number,
        'gemini_answer': gemini_answer_single_language,
        'prompt': prompt_single_language,
        'prompt_method': 'single_language'
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to CSV
df_results.to_csv('../results/mgsm_gemini_results.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"MGSM Gemini runtime is: {elapsed_time} seconds")
