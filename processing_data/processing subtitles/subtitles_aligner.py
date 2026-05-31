import pysrt
import sys
import re
import os

def clean_text(text): # Removes HTML tags, ASS formatting tags, emphasis dots, newlines, and extra spaces
    text = re.sub(r'<[^>]+>', '', text)       
    text = re.sub(r'\{[^}]+\}', '', text)     
    text = text.replace('●', '') 
    text = text.replace('♪', '')             
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'[ ]+', ' ', text)
    text = re.sub(r'^(.+)\s+\1$', r'\1', text) # For duplicates
    text = re.sub(r'\[.*?\]|\(.*?\)|（.*?）|［.*?］|【.*?】', '', text) # For sound effects
    return text

def load(filepath):
    encodings = ['utf-8', 'utf-16', 'utf-8-sig', 'cp1252', 'shift_jis']
    for enc in encodings:
        try:
            subs = pysrt.open(filepath, encoding=enc)
            if len(subs) > 0:
                _ = subs[0]
            return subs
        except UnicodeDecodeError:
            continue
    sys.exit(1)

def align(ja_path, en_path, output_path):    
    ja_subs = load(ja_path)
    en_subs = load(en_path)
    raw_matches = []
    for ja_sub in ja_subs:
        ja_mid = (ja_sub.start.ordinal + ja_sub.end.ordinal) / 2
        best_match = None
        smallest_time_diff = float('inf')
        for en_sub in en_subs:
            if en_sub.text.strip().startswith('<i>') and en_sub.text.strip().endswith('</i>'): # Foreign language
                continue                
            en_mid = (en_sub.start.ordinal + en_sub.end.ordinal) / 2
            time_diff = abs(ja_mid - en_mid)
            if time_diff < smallest_time_diff:
                smallest_time_diff = time_diff
                best_match = en_sub
        if best_match and smallest_time_diff < 2000:
            ja_text = clean_text(ja_sub.text)
            en_text = clean_text(best_match.text)
            if ja_text and en_text:
                raw_matches.append((ja_text, en_text))

    merged_matches = [] # For merging dublicates
    if raw_matches:
        curr_ja, curr_en = raw_matches[0]
        for i in range(1, len(raw_matches)):
            next_ja, next_en = raw_matches[i]
            if next_en == curr_en:
                if not curr_ja.endswith(next_ja):
                    curr_ja += " " + next_ja
            else:
                merged_matches.append((curr_ja, curr_en))
                curr_ja = next_ja
                curr_en = next_en
        merged_matches.append((curr_ja, curr_en))

    with open(output_path, "w", encoding="utf-8") as f:
        for ja_text, en_text in merged_matches:
            f.write(f"{ja_text}\t{en_text}\n")
    print(f"Aligned {len(merged_matches)} sentence pairs.")

if __name__ == "__main__": # Usage: python subtitles_aligner.py <japanese.srt> <english.srt>
    ja_file = sys.argv[1]
    en_file = sys.argv[2]
    out_dir = os.path.dirname(ja_file)
    out_file = os.path.join(out_dir, "aligned_subs.txt")
    align(ja_file, en_file, out_file)