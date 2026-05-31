import os
import sys

def create_chunks(input_path, output_path, chunk_size=15):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    chunks_created = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            f.writelines(chunk)
            f.write("\n" + ("=" * 40) + "\n")
            chunks_created += 1
    print(f"Divided into {chunks_created} chunks.")

if __name__ == "__main__": # Usage: python chunk_corpus.py <aligned_subs.txt>
    input_file = sys.argv[1]
    out_dir = os.path.dirname(input_file)
    output_file = os.path.join(out_dir, "chunked_dialogue_corpus.txt")
    create_chunks(input_file, output_file, chunk_size=15)