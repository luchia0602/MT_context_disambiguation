import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

_MEM_POL_RE = re.compile(r"SPK=[^(]+\([^,]+,(inf|neu|frm)\)")

def predicted_politeness_from_mem(output_line: str) -> str:
    m = _MEM_POL_RE.search(output_line)
    if not m:
        return "neutral"
    tag = m.group(1)
    if tag == "inf":
        return "informal"
    elif tag == "neu":
        return "neutral"
    elif tag == "frm":
        return "formal"
    
    return "neutral"

def process_single_sentence(ja_speaker: str, ja_sentence: str) -> str:
    processor = RealTimeDialogueProcessor()
    return processor.process_turn(ja_speaker, ja_sentence)

df = pd.read_excel("Learning_dataset.xlsx")

plain_df = df[df['politeness_level'] == 'plain'].sample(n=34, random_state=42)
polite_df = df[df['politeness_level'] == 'polite'].sample(n=33, random_state=42)
formal_df = df[df['politeness_level'] == 'formal'].sample(n=33, random_state=42)

sampled_df = pd.concat([plain_df, polite_df, formal_df]).sample(frac=1, random_state=42).reset_index(drop=True)

FORMALITY_TEST_SET = []
label_map = {"plain": "informal", "polite": "neutral", "formal": "formal"}

for _, row in sampled_df.iterrows():
    FORMALITY_TEST_SET.append({
        "label": label_map.get(row['politeness_level'], "neutral"),
        "sentence": str(row['ja_sentence'])
    })

y_true = []
y_pred = []

for sample in FORMALITY_TEST_SET:
    sentence = sample["sentence"]
    true_label = sample["label"]

    output = process_single_sentence("TestSpeaker", sentence)
    pred = predicted_politeness_from_mem(output)

    y_true.append(true_label)
    y_pred.append(pred)

    result = "✓" if pred == true_label else "✗"
    print(f"{result} [{true_label:8s}] pred={pred:8s} | {sentence}")

labels = ["informal", "neutral", "formal"]
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro")

print(f"Accuracy: {accuracy:.3f}")
print(f"Macro F1: {macro_f1:.3f}")
print()
print("Classification Report:")
print()

print(classification_report(y_true, y_pred, labels=labels, digits=3))

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    linecolor="white",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True
)

plt.title("Formality Detection Confusion Matrix", fontsize=15, pad=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

plt.tight_layout()
plt.show()