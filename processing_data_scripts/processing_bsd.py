import json
from tqdm import tqdm

def generate_training_dataset(input_filepath: str, output_filepath: str):
    with open(input_filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    training_data = []

    for dialogue in tqdm(dataset):
        processor = RealTimeDialogueProcessor()

        for index, turn in enumerate(dialogue.get("conversation", [])):
            ja_speaker = turn.get("ja_speaker", "")
            ja_sentence = turn.get("ja_sentence", "")
            en_sentence = turn.get("en_sentence", "")

            raw_output = processor.process_turn(ja_speaker, ja_sentence)

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

    print(f"Successfully processed {len(dataset)} BSD dialogues.")
    print(f"Flattened into {len(training_data)} training pairs.")

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    INPUT_FILE  = "bsd_dev.json"
    OUTPUT_FILE = "mem_bsd_dev.json"
    generate_training_dataset(INPUT_FILE, OUTPUT_FILE)