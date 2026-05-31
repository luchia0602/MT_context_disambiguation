import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def process_manga_dataset(input_filepath: str):
    with open(input_filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    all_pages = []
    for book in dataset:
        book_title = book.get("book_title", "Unknown_Book")
        for page in book.get("pages", []):
            if "text" in page and len(page["text"]) > 0:
                all_pages.append({
                    "book_title": book_title,
                    "page_index": page.get("page_index", 0),
                    "text_bubbles": page["text"]
                })

    print(f"Extracted {len(all_pages)} total dialogues")

    train_pages, temp_pages = train_test_split(all_pages, test_size=0.2, random_state=42)
    dev_pages, test_pages = train_test_split(temp_pages, test_size=0.5, random_state=42)

    print(f"Dataset Split: {len(train_pages)} Train | {len(dev_pages)} Dev | {len(test_pages)} Test")

    def generate_split(pages_list, output_filename):
        training_data = []

        for page in tqdm(pages_list, desc=f"Processing {output_filename}"):
            processor = RealTimeDialogueProcessor()
            valid_turn_index = 0

            for bubble in page["text_bubbles"]:
                ja_sentence = bubble.get("text_ja", "").strip()
                en_sentence = bubble.get("text_en", "").strip()

                if not ja_sentence or not en_sentence:
                    continue

                speaker = "A" if valid_turn_index % 2 == 0 else "B"

                raw_output = processor.process_turn(speaker, ja_sentence)

                mem_tag = ""
                if ": [MEM:" in raw_output:
                    tag_section = raw_output.split(": ", 1)[1]
                    mem_tag = tag_section[:tag_section.find("]") + 1]

                source_text = f"{mem_tag} {en_sentence}".strip()

                if valid_turn_index == 0:
                    source_text = f"[NEW_DIALOGUE] {source_text}"

                training_data.append({
                    "source": source_text,
                    "target": ja_sentence
                })

                valid_turn_index += 1

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(training_data)} training pairs to {output_filename}")

    generate_split(train_pages, "mem_manga_train.json")
    generate_split(dev_pages, "mem_manga_dev.json")
    generate_split(test_pages, "mem_manga_test.json")

if __name__ == "__main__":
    INPUT_FILE = "annotation.json"
    process_manga_dataset(INPUT_FILE)