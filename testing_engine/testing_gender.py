import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report
)

_MR_MS_RE   = re.compile(r"^(Mr\.|Ms\.)\s+", re.IGNORECASE)
_MEM_GEN_RE = re.compile(r"SPK=[^(]+\(([MF?])")


def gender_from_title(en_speaker: str):
    m = _MR_MS_RE.match(en_speaker.strip())
    if not m:
        return None
    return "male" if m.group(1).lower() == "mr." else "female"


def predicted_gender_from_mem(output_line: str):
    m = _MEM_GEN_RE.search(output_line)
    if not m:
        return None

    tag = m.group(1)

    if tag == "M":
        return "male"
    if tag == "F":
        return "female"
    return None

def process_single_sentence(ja_speaker: str, ja_sentence: str) -> str:
    processor = RealTimeDialogueProcessor()
    return processor.process_turn(ja_speaker, ja_sentence)

with open("bsd_train.json", encoding="utf-8") as f:
    raw = json.load(f)
if isinstance(raw, list) and raw and "conversation" in raw[0]:
    turns = []
    for dialogue in raw:
        turns.extend(dialogue["conversation"])
elif isinstance(raw, list):
    turns = raw

y_true = []
y_pred = []

skipped_filter  = 0
skipped_unknown = 0

sources = []

for turn in turns:
    ja_gender   = turn.get("ja_spk_gender", "").strip().upper()
    en_gender   = turn.get("en_spk_gender", "").strip().upper()
    en_speaker  = turn.get("en_speaker", "")
    ja_speaker  = turn.get("ja_speaker", "")
    ja_sentence = turn.get("ja_sentence", "")

    if ja_gender != en_gender or ja_gender not in {"M", "F"}:
        skipped_filter += 1
        continue

    gt_gender = gender_from_title(en_speaker)

    if gt_gender is None:
        skipped_filter += 1
        continue

    corpus_gender = "male" if ja_gender == "M" else "female"

    if gt_gender != corpus_gender:
        skipped_filter += 1
        continue

    output = process_single_sentence(
        ja_speaker,
        ja_sentence
    )

    pred = predicted_gender_from_mem(output)

    if pred is None:
        skipped_unknown += 1
        continue

    y_true.append(corpus_gender)
    y_pred.append(pred)
    sources.append("BSD")

bsd_total   = sum(1 for s in sources if s == "BSD")

print(f"BSD samples evaluated: {bsd_total}")
print(f"BSD skipped (filter): {skipped_filter}")
print(f"BSD unknown predictions: {skipped_unknown}")
print(f"Total evaluated: {len(y_true)}")
print()

for src_name in ["BSD"]:

    idx = [
        i for i, s in enumerate(sources)
        if s == src_name
    ]

    if not idx:
        continue

    t = [y_true[i] for i in idx]
    p = [y_pred[i] for i in idx]

    correct = sum(a == b for a, b in zip(t, p))

    print(
        f"[{src_name}] Accuracy: "
        f"{correct}/{len(t)} = {correct/len(t):.3f}"
    )

print()

labels = ["male", "female"]
print("Classification report:")
print()
print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            digits=3
        )
    )

cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

plt.figure(figsize=(7, 6))

sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=1,
        linecolor="white",
        cbar=True,
        xticklabels=["Male", "Female"],
        yticklabels=["Male", "Female"]
    )

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title(
        "Gender Detection Evaluation on BSD dataset",
        fontsize=14,
        pad=16
    )
plt.tight_layout()
plt.show()

macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

female_f1 = f1_score(
        y_true,
        y_pred,
        pos_label="female",
        average="binary",
        zero_division=0
    )

male_f1 = f1_score(
        y_true,
        y_pred,
        pos_label="male",
        average="binary",
        zero_division=0
    )

total   = len(y_true)
correct = sum(t == p for t, p in zip(y_true, y_pred))

print(f"Macro F1: {macro_f1:.3f}")
print(f"Male F1: {male_f1:.3f}")
print(f"Female F1: {female_f1:.3f}")
print(f"Accuracy: {correct}/{total} = {correct/total:.3f}")
print()