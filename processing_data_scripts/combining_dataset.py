import json
import random
import os


def combine_and_shuffle_datasets():
    datasets = [
        "mem_ami",
        "mem_bsd",
        "mem_manga",
        "mem_meld",
        "mem_persona",
        "mem_subtitles"
    ]

    splits = ["train", "dev", "test"]

    for split in splits:
        combined_data = []
        for ds in datasets:
            file_options = [f"{ds}_{split}.json", f"{ds}_{split}_clean.json"]
            file_found = False

            for filename in file_options:
                if os.path.exists(filename):
                    print(f"Loading {filename}...")
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        combined_data.extend(data)
                        file_found = True
                        break  

        print(f"Shuffling {len(combined_data)} total pairs")
        random.seed(42)  
        random.shuffle(combined_data)

        output_filename = f"combined_ja_en_{split}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    combine_and_shuffle_datasets()