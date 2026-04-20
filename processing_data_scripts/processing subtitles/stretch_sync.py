import pysrt
import sys
import os

def load(filepath):
    encodings = ['utf-8', 'utf-16', 'utf-8-sig', 'cp1252', 'latin-1', 'shift_jis']
    for enc in encodings:
        try:
            subs = pysrt.open(filepath, encoding=enc)
            if len(subs) > 0:
                _ = subs[0]
            return subs
        except UnicodeError: 
            continue
    sys.exit(1)

def stretch(en_path, output_path):
    subs = load(en_path)
    ja_start_ms = 31532
    en_start_ms = 19318
    ja_end_ms = 7695654
    en_end_ms = 7707798
    ja_dur = ja_end_ms - ja_start_ms
    en_dur = en_end_ms - en_start_ms
    stretch_ratio = ja_dur / en_dur
    for sub in subs:
        new_start = ja_start_ms + ((sub.start.ordinal - en_start_ms) * stretch_ratio)
        sub.start.ordinal = max(0, int(new_start))
        new_end = ja_start_ms + ((sub.end.ordinal - en_start_ms) * stretch_ratio)
        sub.end.ordinal = max(0, int(new_end))
    subs.save(output_path, encoding='utf-8')

if __name__ == "__main__": # Usage: python stretch_sync.py <english.srt>
    en_file = sys.argv[1]
    base_name = os.path.splitext(en_file)[0]
    out_file = base_name + "_synced.srt"
    stretch(en_file, out_file)