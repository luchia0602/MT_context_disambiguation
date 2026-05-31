!git clone https://github.com/cl-tohoku/BPersona-chat.git
import os
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_dataset(input_dir: str):
    all_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    train_files, temp_files = train_test_split(all_files, test_size=0.1, random_state=42)
    dev_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    print(f"Dataset Split: {len(train_files)} Train | {len(dev_files)} Dev | {len(test_files)} Test")
    return train_files, dev_files, test_files

def process_xlsx_batch(file_paths: list, output_filepath: str):
    training_data = []

    for file_path in tqdm(file_paths, desc=f"Processing {os.path.basename(output_filepath)}"):
        df = pd.read_excel(file_path)
        processor = RealTimeDialogueProcessor()
        speaker_map = {}
        current_ab = "A"

        for index, row in df.iterrows():
            if pd.isna(row.get('source')) or pd.isna(row.get('translation')):
                continue

            raw_speaker = str(row.get('person', f"unknown_{index}")).strip()
            ja_sentence = str(row.get('source')).strip()
            en_sentence = str(row.get('translation')).strip()

            if not ja_sentence:
                continue

            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = current_ab
                current_ab = "B" if current_ab == "A" else "C"

            speaker_id = speaker_map[raw_speaker]
            raw_output = processor.process_turn(speaker_id, ja_sentence)

            if ": [MEM:" in raw_output:
                source_text = raw_output.split(": ", 1)[1]
            else:
                source_text = raw_output

            if index == 0:
                source_text = f"[NEW_DIALOGUE] {source_text}"

            training_data.append({
                "source": source_text,
                "target": en_sentence
            })

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    DATA_DIR = "BPersona-chat/ja-en/human"
    train_files, dev_files, test_files = split_dataset(DATA_DIR)
    process_xlsx_batch(train_files, "mem_persona_train.json")
    process_xlsx_batch(dev_files, "mem_persona_dev.json")
    process_xlsx_batch(test_files, "mem_persona_test.json")
