!git clone https://github.com/ku-nlp/EaST-MELD.git

import os
import json
from tqdm import tqdm


def process_parallel_dataset(en_filepath: str, ja_filepath: str, output_filepath: str, chunk_size: int = 5):
    with open(en_filepath, 'r', encoding='utf-8') as f_en, \
             open(ja_filepath, 'r', encoding='utf-8') as f_ja:
            en_lines = [line.strip() for line in f_en if line.strip()]
            ja_lines = [line.strip() for line in f_ja if line.strip()]

    paired_lines = list(zip(en_lines, ja_lines))
    chunks = [paired_lines[i:i + chunk_size] for i in range(0, len(paired_lines), chunk_size)]

    training_data = []

    for chunk in tqdm(chunks, desc=f"Processing {os.path.basename(output_filepath)}"):
        processor = RealTimeDialogueProcessor()

        for index, (en_sentence, ja_sentence) in enumerate(chunk):
            speaker = "A" if index % 2 == 0 else "B"
            raw_output = processor.process_turn(speaker, ja_sentence)

            mem_tag = ""
            if ": [MEM:" in raw_output:
                tag_section = raw_output.split(": ", 1)[1]
                mem_tag = tag_section[:tag_section.find("]") + 1]

            source_text = f"{mem_tag} {en_sentence}".strip()

            if index == 0:
                source_text = f"[NEW_DIALOGUE] {source_text}"

            training_data.append({
                "source": source_text,
                "target": ja_sentence
            })

    print(f"Successfully processed {len(paired_lines)} parallel lines into {len(chunks)} chunks.")
    print(f"Saved {len(training_data)} training pairs to {output_filepath}")

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    BASE_DIR = "EaST-MELD/en-ja/txt"

    datasets = [
        {
            "en": os.path.join(BASE_DIR, "train", "train.en"),
            "ja": os.path.join(BASE_DIR, "train", "train.ja"),
            "out": "mem_meld_train.json"
        },
        {
            "en": os.path.join(BASE_DIR, "dev_subtitle", "dev.en"),
            "ja": os.path.join(BASE_DIR, "dev_subtitle", "dev.ja"),
            "out": "mem_meld_dev.json"
        },
        {
            "en": os.path.join(BASE_DIR, "test_subtitle", "test.en"),
            "ja": os.path.join(BASE_DIR, "test_subtitle", "test.ja"),
            "out": "mem_meld_test.json"
        }
    ]

    for ds in datasets:
        process_parallel_dataset(ds["en"], ds["ja"], ds["out"], chunk_size=5)