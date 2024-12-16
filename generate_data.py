import openai
import json
from datasets import load_dataset
import os
import threading
from queue import Queue
from dotenv import load_dotenv
from tqdm import tqdm


def send_question(
    prompt: str,
    model: str,
    api_link: str,
    token: str,
    temperature: float,
    max_tokens: int,
):
    client = openai.OpenAI(
        api_key=token,
        base_url=api_link,
    )

    messages = []
    messages.append({"role": "user", "content": prompt})

    response_big = None
    idx = 0
    while response_big is None or idx == 10:
        response_big = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=1,
            max_tokens=max_tokens,
        )
        idx += 1

    if idx == 10:
        return "This sentence was not translated."

    response = response_big.choices[0].message.content

    return response


def main():
    # Load environment variables
    load_dotenv()

    # Load the dataset
    data = load_dataset("lmsys/lmsys-chat-1m")

    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    API_LINK = "https://openrouter.ai/api/v1"
    OUTPUT_FILE = "generated_data.json"
    LANGUAGES = ["English", "Russian", "German", "French", "Spanish"]
    SENTENCES_PER_LANGUAGE = 2000

    MODELS = [
        {"name": "google/gemini-flash-1.5", "answer_column": "answer_1", "refusal_column": "refusal_1"},
        {"name": "meta-llama/llama-3.3-70b-instruct", "answer_column": "answer_2", "refusal_column": "refusal_2"}
    ]


    def process_entry(entry, lang, output_rows):
        # Extract input from conversation
        conversation = entry["conversation"]
        if not conversation or len(conversation) < 2:
            return

        input_text = conversation[0]['content']  # First input

        # Ensure the input_text has more than 5 words
        if len(input_text.split()) < 5:
            return  # Skip this entry if the condition is not met

        row = {"input": input_text}

        for model in MODELS:
            # Generate an answer
            answer_prompt = f"{input_text}"
            answer = send_question(
                prompt=answer_prompt,
                model=model["name"],
                api_link=API_LINK,
                token=API_KEY,
                temperature=0.7,
                max_tokens=200
            )

            # Generate a refusal
            refusal_prompt = f"Politely refuse to answer this in {lang} and provide an explanation why you refuse. The refusal should be connected to the request topic. Do not add anything additional, only respond with a refusal: {input_text}"
            refusal = send_question(
                prompt=refusal_prompt,
                model=model["name"],
                api_link=API_LINK,
                token=API_KEY,
                temperature=0.7,
                max_tokens=200
            )

            if answer and refusal:
                row[model["answer_column"]] = answer
                row[model["refusal_column"]] = refusal

        if len(row) == 5:  # Ensure all data is present
            output_rows.append(row)


    def process_entries(lang, lang_data, output_file):
        print(f"Processing language: {lang}")
        output_rows = []
        queue = Queue()

        def worker():
            while not queue.empty():
                entry = queue.get()
                try:
                    process_entry(entry, lang, output_rows)
                except Exception as e:
                    print(f"Error processing entry: {e}")
                finally:
                    queue.task_done()

        lang_data_iterator = filter(lambda conv: conv["language"] == lang, data["train"])  # Use lazy evaluation
        for _ in tqdm(range(SENTENCES_PER_LANGUAGE), desc=f"Preparing queue for {lang}"):
            try:
                entry = next(lang_data_iterator)
                queue.put(entry)
            except StopIteration:
                break

        threads = []
        for _ in range(10):  # 10 threads for processing
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        queue.join()
        for thread in threads:
            thread.join(timeout=10)  # Timeout to prevent deadlock
            if thread.is_alive():
                print("Warning: A thread failed to complete within the timeout.")

        # Save results for the current language
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=4)

        print(f"Completed processing for {lang}.")


    for lang in tqdm(LANGUAGES, desc="Processing all languages"):
        output_file = f"results/{lang.lower()}_data.json"
        process_entries(lang, data["train"], output_file)

    print(f"Generated data saved for all languages.")


if __name__ == "__main__":
    main()

