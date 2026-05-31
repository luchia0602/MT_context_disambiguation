!pip install sudachipy sudachidict-core pykakasi gender-guesser gensim
import os
import re
import math
import bz2
import urllib.request
import numpy as np
import pykakasi
import gender_guesser.detector as gender
from sudachipy import dictionary, tokenizer
from gensim.models import KeyedVectors

MODEL_TXT_BZ2_URL = (
    "https://github.com/singletongue/WikiEntVec/releases/download/"
    "20190520/jawiki.word_vectors.200d.txt.bz2"
)
MODEL_TXT = "jawiki.word_vectors.200d.txt"
MODEL_KV  = "jawiki.kv"

if 'w2v' not in globals():
    if not os.path.exists(MODEL_KV):
        print(f"Model [{MODEL_KV}] was not found: downloading...")
        if not os.path.exists(MODEL_TXT):
            print(f"Downloading jawiki word vectors from {MODEL_TXT_BZ2_URL}...")
            urllib.request.urlretrieve(MODEL_TXT_BZ2_URL, MODEL_TXT + ".bz2")
            with bz2.BZ2File(MODEL_TXT + ".bz2", "rb") as source, open(MODEL_TXT, "wb") as dest:
                dest.write(source.read())
            os.remove(MODEL_TXT + ".bz2")
        tmp_w2v = KeyedVectors.load_word2vec_format(MODEL_TXT, binary=False)
        tmp_w2v.save(MODEL_KV)
        print("Download complete")
    print("Loading Word2Vec model...")
    w2v = KeyedVectors.load(MODEL_KV, mmap="r")
else:
    print("Word2Vec model already loaded")

tok = dictionary.Dictionary().create()

ADDRESS_SUFFIXES  = {"さん", "くん", "ちゃん", "先生", "様", "殿", "部長", "課長", "氏", "社長"}
GENERIC_SUFFIXES  = {"さん", "くん", "ちゃん", "様", "殿", "氏"}

FIRST_PERSON_PRONOUNS = {
    "私", "わたし", "わたくし", "あたくし", "俺", "僕",
    "あたし", "わし", "自分", "俺ら", "僕ら",
}
SECOND_PERSON_PRONOUNS = {
    "あなた", "あんた", "きみ", "君", "お前", "貴様",
    "てめえ", "てめ", "お前さん", "あなた様",
}
EXPLICIT_PRONOUNS_TO_RESOLVE = {"彼", "彼女", "彼ら", "彼女たち", "こいつ", "そいつ", "あいつ", "奴"}

CASE_PARTICLE_ROLE = {
    "は": "topic", "が": "subject", "を": "object",
    "に": "indirect_object", "も": "topic",
    "から": "indirect_object", "まで": "indirect_object",
    "より": "indirect_object", "へ": "indirect_object",
}

CONDITIONAL_TOPIC_RE = re.compile(
    r"(でしたら|であれば|ならば|なら|とすれば|とすると|だったら)$"
)

DESIDERATIVE_MORPHEMES = {"たい", "たがる", "たがっている"}
LISTENER_MORPHEMES = {
    "てください", "でください", "ください", "なさい", "てくれ", "でくれ",
    "てくれる", "でくれる", "ていい", "でいい", "てもいい", "でもいい",
    "なくていい", "てくれない", "でくれない", "てもらえない", "でもらえない",
    "てほしい", "でほしい",
}
SPEAKER_MORPHEMES = {
    "てあげる", "であげる", "てさしあげる", "でさしあげる", "ておく", "でおく",
    "てしまう", "でしまう", "てみる", "でみる", "てみせる", "でみせる",
}

LIGHTWEIGHT_VALENCY_VETO = {
    "頼む", "承知する", "お願いする", "いたす", "承知いたす", "話す", "言う", "語る",
}
INTRANSITIVE_VERBS = {
    "動く", "出る", "終わる", "始まる", "済む", "進む", "変わる",
    "行く", "来る", "帰る", "わかる", "着く", "見える", "聞こえる",
    "違う", "なる", "ある", "いる", "咲く", "降る", "落ちる",
}

IDIOM_BLACKLIST = {"世話", "迷惑", "お陰", "おかげ", "お世話", "いつもお世話", "世話様"}

FORMAL_DICTIONARY_FORMS = {
    "ござる", "致す", "申す", "頂く", "伺う",
    "参る", "おっしゃる", "なさる", "くださる", "存じる",
    "召し上がる", "おる", "賜る",
}

NEUTRAL_DICTIONARY_FORMS = {
    "ます", "です"
}

HONORIFIC_PREFIX_PATTERN = re.compile(r"^[おごご御]")

PRONOUN_FEATURES = {
    "彼女":     {"animate", "human", "female"},
    "彼":       {"animate", "human", "male"},
    "私":       {"animate", "human"},
    "わたし":   {"animate", "human"},
    "わたくし": {"animate", "human"},
    "あたくし": {"animate", "human", "female"},
    "あなた":   {"animate", "human"},
    "あんた":   {"animate", "human"},
    "俺":       {"animate", "human", "male"},
    "僕":       {"animate", "human", "male"},
    "あたし":   {"animate", "human", "female"},
    "わし":     {"animate", "human", "male"},
    "自分":     {"animate", "human"},
    "きみ":     {"animate", "human"},
    "君":       {"animate", "human"},
    "お前":     {"animate", "human"},
    "貴様":     {"animate", "human"},
    "てめえ":   {"animate", "human"},
    "てめ":     {"animate", "human"},
    "お前さん": {"animate", "human"},
    "あなた様": {"animate", "human"},
    "こちら":   {"animate", "human"},
    "そちら":   {"animate", "human"},
    "彼ら":     {"animate", "human", "male", "plural"},
    "彼女たち": {"animate", "human", "female", "plural"},
}

PLURAL_SUFFIXES = re.compile(r"(たち|ども|ら)$")

SUDACHI_POS_TO_FEATURES = {
    "人名": {"animate", "human"},
    "地名": {"location"},
    "動物": {"animate"},
}

LIGHT_VERBS = {"する", "いたす", "できる", "ある", "いる", "なる"}

def get_top_k_distribution(score_breakdown: dict, k: int = 3) -> dict:
    if not score_breakdown:
        return {}
    valid = {e: d["total"] for e, d in score_breakdown.items() if d["total"] > 0}
    if not valid:
        return {}
    top   = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:k]
    total = sum(s for _, s in top)
    return {e: round(s / total, 2) for e, s in top}

PROTO_ANIMATE_WORDS   = ["人", "男", "女", "彼", "彼女", "子供", "社員", "先生", "友達", "客"]
PROTO_INANIMATE_WORDS = ["物", "書類", "会議", "仕事", "問題", "計画", "資料", "情報", "製品", "結果"]

def _mean_vec(words):
    vecs = [w2v[w] for w in words if w in w2v]
    return np.mean(vecs, axis=0) if vecs else None

_PROTO_ANIMATE_VEC   = _mean_vec(PROTO_ANIMATE_WORDS)
_PROTO_INANIMATE_VEC = _mean_vec(PROTO_INANIMATE_WORDS)

def _get_vec(word):
    return w2v[word] if word in w2v else None

def _cosine(a, b) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def _sigmoid_norm(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-5.0 * x))

def animacy_score_w2v(candidate_text: str) -> float:
    if _PROTO_ANIMATE_VEC is None or _PROTO_INANIMATE_VEC is None:
        return 0.5
    lookup = PLURAL_SUFFIXES.sub("", candidate_text)
    for suf in ADDRESS_SUFFIXES:
        lookup = lookup.replace(suf, "")
    lookup = lookup.strip() or candidate_text

    v = _get_vec(lookup)
    if v is None:
        v = _get_vec(candidate_text)

    if v is None:
        return 0.5

    return _sigmoid_norm(_cosine(v, _PROTO_ANIMATE_VEC) - _cosine(v, _PROTO_INANIMATE_VEC))

def parse_verb_morphology(sentence: str) -> dict:
    tokens = tok.tokenize(sentence, tokenizer.Tokenizer.SplitMode.A)
    result = {
        "main_verb_dict": "", "aux_verbs": [],
        "entails_animate": False, "entails_human": False,
        "entails_speaker": False, "entails_listener": False,
        "is_volitional": False, "is_desiderative": False,
        "is_inanimate_event": False, "is_intransitive": False,
    }
    surfaces    = [m.surface() for m in tokens]
    dict_forms  = [m.dictionary_form() for m in tokens]
    pos_list    = [m.part_of_speech() for m in tokens]
    surface_str = "".join(surfaces)

    verb_pos = [p for p in pos_list if p[0] == "動詞"]
    if verb_pos and len(verb_pos[-1]) > 4 and "自動詞" in verb_pos[-1][4]:
        result["is_intransitive"] = True

    vdf = [dict_forms[i] for i, p in enumerate(pos_list) if p[0] == "動詞"]
    if vdf:
        result["main_verb_dict"] = vdf[-1]
        result["aux_verbs"]      = vdf[:-1]

    if any(s in surface_str for s in {"させられ", "せられ"}):
        result["entails_animate"] = result["entails_human"] = True

    if "ましょうか" in surface_str:
        result["entails_animate"] = result["entails_human"] = True
        result["is_volitional"]   = result["entails_speaker"] = True
    elif any(s in surface_str for s in {"ましょう", "ませんか", "ようか", "ようと", "よう"}):
        result["entails_animate"] = result["entails_human"] = result["is_volitional"] = True

    if any(s in surface_str for s in DESIDERATIVE_MORPHEMES):
        result["entails_animate"] = result["entails_human"] = True
        result["is_desiderative"] = result["entails_speaker"] = True

    if any(s in surface_str for s in LISTENER_MORPHEMES):
        result["entails_animate"] = result["entails_human"] = result["entails_listener"] = True

    if any(s in surface_str for s in SPEAKER_MORPHEMES):
        result["entails_animate"] = result["entails_human"] = result["entails_speaker"] = True

    if any(d in FORMAL_DICTIONARY_FORMS for d in dict_forms):
        result["entails_animate"] = result["entails_human"] = True

    if not result["entails_animate"] and not result["entails_human"]:
        for i, p in enumerate(pos_list):
            if p[0] == "動詞" and len(p) > 4 and "自動詞" in p[4]:
                result["is_inanimate_event"] = True
                break

    return result

def combined_semantic_score(candidate_features: set, candidate_text: str, morph: dict, zero_role: str = "subject") -> float:
    if zero_role in {"subject", "topic"}:
        if morph["is_inanimate_event"]:
            morph_score = 0.1 if "human" in candidate_features else 0.9
        elif morph["entails_human"]:
            morph_score = 1.0 if "human" in candidate_features else 0.0
        elif morph["entails_animate"]:
            morph_score = 1.0 if "animate" in candidate_features else 0.0
        else:
            morph_score = 0.5
    else:
        if morph["entails_listener"]:
            morph_score = 1.0 if "human" in candidate_features else 0.1
        elif "human" in candidate_features or "animate" in candidate_features:
            morph_score = 0.2
        else:
            morph_score = 0.8

    proto_score = animacy_score_w2v(candidate_text)

    if morph_score != 0.5:
        return 0.85 * morph_score + 0.15 * proto_score

    if "human" in candidate_features:     feature_score = 1.0
    elif "animate" in candidate_features: feature_score = 0.8
    elif "location" in candidate_features:feature_score = 0.2
    else:                                 feature_score = 0.5

    if zero_role == "object":
        if feature_score >= 0.8: feature_score = 0.2
        elif feature_score <= 0.5: feature_score = 0.8

    return 0.55 * feature_score + 0.45 * proto_score

def object_drop_warranted(morph: dict) -> bool:
    mv = morph.get("main_verb_dict", "")
    if mv in LIGHTWEIGHT_VALENCY_VETO or mv in INTRANSITIVE_VERBS or morph.get("is_intransitive"):
        return False
    return True

def infer_semantic_features(entity_text: str, sudachi_pos_tags: list) -> set:
    if entity_text in PRONOUN_FEATURES:
        return set(PRONOUN_FEATURES[entity_text])
    features = set()
    for tag in sudachi_pos_tags:
        if tag in SUDACHI_POS_TO_FEATURES:
            features |= SUDACHI_POS_TO_FEATURES[tag]
    return features

W_CENTERING = 0.20
W_PROPERTY = 0.20
W_SEMANTIC = 0.50
W_CONTINUITY = 0.30
SPEAKER_BOOST = 0.40
RECENCY_DECAY = 0.85

_MAX_RESOLVER_SCORE = W_CENTERING + W_PROPERTY + W_SEMANTIC + W_CONTINUITY * 2.5 + SPEAKER_BOOST

ROLE_SALIENCE = {
    "topic": 1.0, "subject": 0.8, "object": 0.5, "indirect_object": 0.3,
    "modifier": 0.2, "speaker_ref": 0.2, "predicate": 0.1, "action": 0.1,
}

def centering_score(candidate: dict, stack: list,
                    speaker_label: str = "", listener_labels: set = None) -> float:
    if listener_labels is None:
        listener_labels = set()
    if candidate["entity"] == speaker_label or candidate["entity"] in listener_labels:
        return 1.0
    role_score = ROLE_SALIENCE.get(candidate["role"], 0.1)
    depth = next(
        (i for i, item in enumerate(reversed(stack)) if item["entity"] == candidate["entity"]),
        len(stack),
    )
    return role_score * (RECENCY_DECAY ** depth)

def property_sharing_score(candidate_role: str, zero_role: str) -> float:
    if zero_role in {"subject", "topic"} and candidate_role in {"subject", "topic", "speaker_ref"}:
        return 1.0
    if zero_role == "object" and candidate_role == "object":
        return 1.0
    if zero_role in {"subject", "topic"} and candidate_role == "object":
        return 0.7
    return 0.2

def normalise_resolver_confidence(best_score: float, score_breakdown: dict) -> float:
    if not score_breakdown:
        return 0.0
    scores    = sorted([v["total"] for v in score_breakdown.values()], reverse=True)
    base_conf = min(best_score, _MAX_RESOLVER_SCORE) / _MAX_RESOLVER_SCORE
    if len(scores) >= 2:
        margin        = scores[0] - scores[1]
        margin_factor = min(margin / 0.3, 1.0)
        base_conf     = base_conf * (0.5 + 0.5 * margin_factor)
    return round(min(max(base_conf, 0.0), 1.0), 2)

def explicit_pronoun_confidence(candidate: dict, stack: list,
                                current_speaker_label: str,
                                listener_labels: set) -> float:
    FLOOR = 0.55
    HEADROOM = 0.88 - FLOOR

    if candidate.get("role") == "speaker_ref":
        cs = centering_score(candidate, stack, current_speaker_label, listener_labels)
        return round(min(max(FLOOR + HEADROOM * cs, FLOOR), 0.88), 2)

    stored_conf   = candidate.get("conf", 1.0)
    role_salience = ROLE_SALIENCE.get(candidate.get("role", "predicate"), 0.1)
    depth = next(
        (i for i, item in enumerate(reversed(stack)) if item["entity"] == candidate["entity"]),
        len(stack),
    )
    score = FLOOR + HEADROOM * stored_conf * role_salience * (0.92 ** depth)
    return round(min(max(score, FLOOR), 0.88), 2)

def np_confidence(entity_text: str, features: set) -> float:
    if "human" in features:
        return 1.0
    if "animate" in features:
        return 0.96
    w2v_score = animacy_score_w2v(entity_text)
    if w2v_score == 0.5:
        return 0.85
    inanimate_strength = 1.0 - w2v_score
    return round(0.82 + 0.11 * inanimate_strength, 2)

def resolve_zero_pronoun(stack, zero_role, morph: dict, exclude_entities,
                         entity_features, speaker_aliases_map,
                         last_active_subject: str = "",
                         last_active_object: str = "",
                         current_speaker_label: str = "",
                         listener_labels: set = None,
                         is_question: bool = False,
                         is_imperative: bool = False,
                         is_permission: bool = False):

    if listener_labels is None:
        listener_labels = set()

    seen = {}
    for item in stack:
        if item["entity"] in exclude_entities or item["role"] == "action":
            continue
        entity = item["entity"]
        if entity not in seen:
            seen[entity] = item
        elif ROLE_SALIENCE.get(item["role"], 0.1) > ROLE_SALIENCE.get(seen[entity]["role"], 0.1):
            seen[entity] = item

    if zero_role in {"subject", "topic"}:
        if morph.get("is_inanimate_event"):
            inanimate_cands = [
                c for c in seen.values()
                if "human" not in entity_features.get(c["entity"], c.get("features", set()))
                and c["entity"] != current_speaker_label
                and c["entity"] not in listener_labels
            ]
            if inanimate_cands:
                best = max(inanimate_cands, key=lambda c: centering_score(
                    c, stack, current_speaker_label, listener_labels))
                return best, {best["entity"]: {"total": 9.99}}, 1.0

        if morph.get("is_desiderative") or morph.get("entails_speaker"):
            if current_speaker_label in seen:
                return seen[current_speaker_label], {current_speaker_label: {"total": 9.99}}, 1.0

        if morph.get("entails_listener") or is_permission or is_imperative:
            if listener_labels:
                for l_label in listener_labels:
                    if l_label in seen:
                        return seen[l_label], {l_label: {"total": 9.99}}, 1.0
                return {"entity": list(listener_labels)[0], "role": "speaker_ref", "conf": 1.0}, {list(listener_labels)[0]: {"total": 9.99}}, 1.0
            else:
                return {"entity": "Unknown_Listener", "role": "speaker_ref", "conf": 1.0}, {"Unknown_Listener": {"total": 9.99}}, 1.0

    candidates = list(seen.values())
    if not candidates:
        return None, {}, 0.0

    best, best_score = None, -1.0
    score_breakdown  = {}

    for candidate in candidates:
        cs = centering_score(candidate, stack, current_speaker_label, listener_labels)
        ps = property_sharing_score(candidate["role"], zero_role)
        feats = entity_features.get(candidate["entity"], candidate.get("features", set()))
        sel = combined_semantic_score(feats, candidate["entity"], morph, zero_role)

        cont = 0.0
        if zero_role in {"subject", "topic"} and candidate["entity"] == last_active_subject:
            cont = 2.0
        elif zero_role == "object":
            if candidate["entity"] == last_active_object:
                cont = 2.0
            elif candidate["entity"] == last_active_subject:
                fsub = entity_features.get(candidate["entity"], candidate.get("features", set()))
                if "human" not in fsub and "animate" not in fsub:
                    if animacy_score_w2v(last_active_subject) <= 0.55:
                        cont = 2.5

        total = W_CENTERING * cs + W_PROPERTY * ps + W_SEMANTIC * sel + W_CONTINUITY * cont
        score_breakdown[candidate["entity"]] = {
            "centering":  round(cs,    3),
            "property":   round(ps,    3),
            "semantic":   round(sel,   3),
            "continuity": round(cont,  3),
            "total":      round(total, 3),
        }
        if total > best_score:
            best_score = total
            best       = candidate

    if zero_role in {"subject", "topic"}:
        target_participants = listener_labels if is_question else {current_speaker_label}
        for tp in target_participants:
            if tp in seen:
                candidate = seen[tp]
                cs    = centering_score(candidate, stack, current_speaker_label, listener_labels)
                ps    = property_sharing_score(candidate["role"], zero_role)
                feats = entity_features.get(candidate["entity"], candidate.get("features", set()))
                sel   = combined_semantic_score(feats, candidate["entity"], morph, zero_role)
                cont  = 2.0 if candidate["entity"] == last_active_subject else 0.0

                scaled_boost = SPEAKER_BOOST * sel
                spk_total = (W_CENTERING * cs + W_PROPERTY * ps + W_SEMANTIC * sel
                             + W_CONTINUITY * cont + scaled_boost)

                if spk_total > best_score:
                    best       = seen[tp]
                    best_score = spk_total

                score_breakdown[tp] = {
                    "centering": round(cs, 3),
                    "property": round(ps, 3),
                    "semantic": round(sel, 3),
                    "continuity": round(cont, 3),
                    "total": round(spk_total, 3),
                }

    confidence = normalise_resolver_confidence(best_score, score_breakdown)
    return best, score_breakdown, confidence

class GenderResolver:
    def __init__(self):
        self.kks     = pykakasi.kakasi()
        self.guesser = gender.Detector()

    def get_gender(self, name: str) -> str:
        if not name: return "none"
        if name.endswith("くん"):   return "male"
        if name.endswith("ちゃん"): return "female"
        res    = self.kks.convert(name)
        romaji = "".join(item["hepburn"] for item in res).strip().capitalize()
        first  = romaji.split()[0] if romaji else ""
        guess  = self.guesser.get_gender(first)
        return ("female" if guess in ("female", "mostly_female")
                else "male" if guess in ("male", "mostly_male")
                else "none")

gender_resolver = GenderResolver()

POLITENESS_ALPHA   = 0.6
_POLITENESS_LEVELS = {"informal": 0.0, "neutral": 0.5, "formal": 1.0}
_POLITENESS_INV    = {0.0: "informal", 0.5: "neutral", 1.0: "formal"}

FORMAL_PRONOUNS = {"わたくし", "あたくし", "自分", "あなた様"}
NEUTRAL_PRONOUNS = {"わたし", "あなた", "おたく", "私"}
INFORMAL_PRONOUNS = {"俺", "僕", "あたし", "わし", "あんた", "きみ", "君", "お前", "貴様", "てめえ", "てめ", "お前さん"}

FORMAL_SUFFIXES = {"様", "氏", "殿", "社長", "部長", "課長", "先生"}
NEUTRAL_SUFFIXES = {"さん"}
INFORMAL_SUFFIXES = {"くん", "ちゃん"}

def update_politeness(speaker, new_level: str) -> None:
    cur     = _POLITENESS_LEVELS.get(speaker.politeness, 0.0)
    new     = _POLITENESS_LEVELS.get(new_level, 0.0)
    smooth  = (1 - POLITENESS_ALPHA) * cur + POLITENESS_ALPHA * new
    best_k  = min(_POLITENESS_INV, key=lambda k: abs(k - smooth))
    speaker.politeness = _POLITENESS_INV[best_k]

def score_politeness(sentence: str) -> str:
    tokens = tok.tokenize(sentence, tokenizer.Tokenizer.SplitMode.A)
    scores = {"formal": 0, "neutral": 0, "informal": 0}
    for m in tokens:
        s, d, pos = m.surface(), m.dictionary_form(), m.part_of_speech()
        if d in FORMAL_DICTIONARY_FORMS:
            scores["formal"] += 2
        elif d in NEUTRAL_DICTIONARY_FORMS:
            scores["neutral"] += 2
            
        if pos[0] in {"名詞", "固有名詞"} and HONORIFIC_PREFIX_PATTERN.match(s):
            scores["formal"] += 1
            
        if s in FORMAL_PRONOUNS:
            scores["formal"] += 2
        elif s in NEUTRAL_PRONOUNS:
            scores["neutral"] += 1
        elif s in INFORMAL_PRONOUNS:
            scores["informal"] += 2
            
        if s in FORMAL_SUFFIXES:
            scores["formal"] += 2
        elif s in NEUTRAL_SUFFIXES:
            scores["neutral"] += 1
        elif s in INFORMAL_SUFFIXES:
            scores["informal"] += 2

    mx = max(scores.values())
    if mx == 0: return "informal"
    
    for level in ("formal", "neutral", "informal"):
        if scores[level] == mx: return level

class SpeakerMemory:
    def __init__(self, sid):
        self.sid = sid
        self.name = None
        self.aliases = set()
        self.gender = "none"
        self.politeness = "informal"

    def label(self) -> str:
        return self.name if self.name else f"Speaker_{self.sid}"

class DialogueMemory:
    def __init__(self, speaker_ids: list, registry: dict):
        self.stack               = []
        self.entity_features     = {}
        self.last_active_subject = ""
        self.last_active_object  = ""
        self._registry           = registry
        self._sid_to_label       = {sid: f"Speaker_{sid}" for sid in speaker_ids}

    def update_speaker_label(self, sid: str, new_label: str,
                              new_gender: str, title: str = "") -> None:
        speaker   = self._registry[sid]
        old_label = self._sid_to_label.get(sid, f"Speaker_{sid}")
        if speaker.name is None:
            speaker.name = new_label
            speaker.aliases.add(new_label)
            if title and title not in GENERIC_SUFFIXES:
                speaker.aliases.add(title)
                speaker.aliases.add(new_label + title)
            self._sid_to_label[sid] = new_label
            for item in self.stack:
                if item["entity"] == old_label and item["role"] == "speaker_ref":
                    item["entity"] = new_label
                    if new_gender != "none": item["gender"] = new_gender
                    break
            if old_label in self.entity_features and old_label != new_label:
                self.entity_features[new_label] = self.entity_features.pop(old_label)
            if self.last_active_subject == old_label: self.last_active_subject = new_label
            if self.last_active_object  == old_label: self.last_active_object  = new_label
        else:
            speaker.aliases.add(new_label)
            if title and title not in GENERIC_SUFFIXES:
                speaker.aliases.add(title)
                speaker.aliases.add(new_label + title)
        if new_gender != "none" and speaker.gender == "none":
            speaker.gender = new_gender

    def reset_continuity(self) -> None:
        self.last_active_subject = ""

    def resolve_explicit_pronoun(self, pronoun_dict: dict,
                                  current_speaker_label: str,
                                  listener_labels: set):
        candidates = [
            c for c in self.stack
            if c["entity"] not in {"[ZERO_SUBJECT]", "[ZERO_OBJECT]"}
            and c["role"] != "action"
            and c["entity"] != current_speaker_label
        ]
        p_gen = pronoun_dict.get("gender", "none")
        if p_gen != "none":
            candidates = [
                c for c in candidates
                if (c.get("gender", "none") == p_gen or c.get("gender", "none") == "none")
                and "human" in c.get("features", self.entity_features.get(c["entity"], set()))
            ]
        p_num = pronoun_dict.get("number", "singular")
        if p_num != "singular":
            candidates = [c for c in candidates if c.get("number", "singular") == p_num]
        non_participant = [
            c for c in candidates
            if c["entity"] != current_speaker_label and c["entity"] not in listener_labels
        ]
        if not non_participant:
            return None, 0.0
        best = max(non_participant, key=lambda c: centering_score(
            c, self.stack, current_speaker_label, listener_labels))
        if p_gen != "none" and best.get("gender", "none") == "none":
            best["gender"] = p_gen
        conf = explicit_pronoun_confidence(best, self.stack, current_speaker_label, listener_labels)
        return best, conf

    def update(self, entities: list) -> None:
        for e in entities:
            if e["original_entity"] in ("[ZERO_SUBJECT]", "[ZERO_OBJECT]"):
                continue
            entity = e["entity"]
            if not entity:
                continue
            conf = e.get("conf", 1.0)
            spk_idx = next(
                (i for i, item in enumerate(self.stack)
                 if item["entity"] == entity and item["role"] == "speaker_ref"),
                None,
            )
            if spk_idx is not None:
                item = self.stack[spk_idx]
                if e.get("gender", "none") != "none": item["gender"] = e["gender"]
                if e.get("number", "singular") == "plural": item["number"] = e["number"]
            else:
                self.stack = [item for item in self.stack if item["entity"] != entity]
                new_entry  = {"entity": entity, "role": e["role"], "conf": conf}
                if e.get("gender", "none") != "none": new_entry["gender"]   = e["gender"]
                if e.get("number", "singular") == "plural": new_entry["number"] = e["number"]
                if e.get("features"):                new_entry["features"] = e["features"]
                self.stack.append(new_entry)
            if e.get("features"):
                self.entity_features[entity] = e["features"]

    def resolve(self, zero_role, morph, exclude_entities,
                current_speaker_label="", listener_labels=None,
                is_question=False, is_imperative=False,
                is_permission=False):
        if listener_labels is None: listener_labels = set()
        return resolve_zero_pronoun(
            self.stack, zero_role, morph, exclude_entities,
            self.entity_features,
            {s.label(): s.aliases for s in self._registry.values()},
            last_active_subject   = self.last_active_subject,
            last_active_object    = self.last_active_object,
            current_speaker_label = current_speaker_label,
            listener_labels       = listener_labels,
            is_question           = is_question,
            is_imperative         = is_imperative,
            is_permission         = is_permission,
        )

def classify_no(tokens_list: list, idx: int) -> str:
    if idx + 1 >= len(tokens_list): return "sentence_final"
    nt   = tokens_list[idx + 1]
    npos = nt.part_of_speech()[0]
    ns   = nt.surface()
    if npos in {"名詞", "代名詞", "固有名詞", "接尾辞"}:          return "possessive"
    if npos in {"動詞", "助動詞", "形容詞", "形容動詞"}:            return "nominalizer"
    if ns in {"か", "に", "で", "が", "を", "は", "も", "へ", "と", "から", "まで"}:
        return "nominalizer"
    return "sentence_final"

_TEMPORAL_ADVERBIALS = {
    "もう", "まだ", "後", "後で", "今", "今から", "今後", "先", "先ほど",
    "昨日", "今日", "明日", "朝", "夜", "午前", "午後", "最近", "今回",
    "今度", "前回", "毎回", "来週", "先週", "今週", "毎週", "来月", "先月",
    "今月", "毎月", "来年", "去年", "今年", "毎年", "いつも", "たまに",
    "すぐ", "すぐに", "早速", "改めて", "引き続き", "続いて",
}
_QUESTION_WORDS = {
    "どこ", "何", "誰", "いつ", "どれ", "どの", "どちら", "どう", "なぜ",
    "どうして", "何で", "どんな", "どのくらい", "どのように", "いくら", "いくつ",
}
_DISCOURSE_FILLERS = {
    "はい", "ええ", "うん", "いや", "いいえ", "あ", "あの", "えー", "えっ",
    "まあ", "では", "じゃ", "じゃあ", "さて", "ところで", "それで", "なお",
    "ただし", "ちなみに", "要するに", "つまり", "すなわち", "むしろ",
}
_GREETING_PREDICATES = {
    "お疲れ", "お疲れ様", "お疲れさま", "ご苦労", "ご苦労様",
    "よろしく", "よろしくお願い", "失礼", "お邪魔",
    "すみません", "すいません", "ごめんなさい", "ごめん", "ありがとう", "ありがとうございます",
}
_SINGLE_CHAR_FUNCTION_NOUNS = {
    "気", "等", "分", "点", "方", "側", "間", "中", "上", "下",
    "前", "後", "内", "外", "他", "頃", "際", "度", "番", "号",
}

_TEMPORAL_REGEX = re.compile(r"^[約]?\s*[一二三四五六七八九十百千万０-９0-9]+(秒|分|時間|日|週|ヶ?月|年)(半|ほど|くらい|ぐらい)?(前|後|間)?$")

def is_stack_worthy(text: str, role: str, sudachi_pos_tags: list) -> bool:
    if text in _TEMPORAL_ADVERBIALS:  return False
    if text in _QUESTION_WORDS:       return False
    if text in _DISCOURSE_FILLERS:    return False
    if text in _GREETING_PREDICATES:  return False
    if _TEMPORAL_REGEX.search(text):  return False
    if text in _SINGLE_CHAR_FUNCTION_NOUNS and len(sudachi_pos_tags) <= 1: return False
    if len(text) <= 2 and re.fullmatch(r"[ぁ-ん]+", text): return False
    if role == "predicate":
        has_proper  = any("固有名詞" in t for t in sudachi_pos_tags)
        has_animate = any(t in SUDACHI_POS_TO_FEATURES and
                          "animate" in SUDACHI_POS_TO_FEATURES[t]
                          for t in sudachi_pos_tags)
        if not has_proper and not has_animate: return False
    return True

def should_flush_before(morpheme, buffer) -> bool:
    pos = morpheme.part_of_speech()
    if pos[0] == "数詞": return True
    if "助数詞" in pos:
        last = buffer[-1] if buffer else ""
        return not bool(re.search(r"[一二三四五六七八九十百千万億兆\d]", last))
    if "時相名詞" in pos: return True
    return False

def create_entity_dict(text: str, role: str, sudachi_pos_tags: list):
    if text in IDIOM_BLACKLIST or not sudachi_pos_tags:
        return None
    head_tag = sudachi_pos_tags[-1]
    if "副詞" in head_tag or "接続詞" in head_tag: return None
    if "時相名詞" in head_tag and len(sudachi_pos_tags) == 1: return None

    gate_text = text
    if not is_stack_worthy(gate_text, role, sudachi_pos_tags): return None

    num      = "plural" if bool(re.search(r"[二三四五六七八九十]|たち|ら", text)) else "singular"
    features = infer_semantic_features(text, sudachi_pos_tags)
    is_proper = (
        any("固有名詞" in tag for tag in sudachi_pos_tags)
        or any(text.endswith(s) for s in ADDRESS_SUFFIXES)
    )
    if "male" in features:      gen = "male"
    elif "female" in features:  gen = "female"
    elif is_proper:             gen = gender_resolver.get_gender(text)
    else:                       gen = "none"

    clean = text
    for s in ADDRESS_SUFFIXES: clean = clean.replace(s, "")
    if not clean.strip(): return None

    conf  = np_confidence(clean, features)
    entry = {"original_entity": text, "entity": clean, "role": role, "conf": conf}
    if gen != "none":   entry["gender"]   = gen
    if num == "plural": entry["number"]   = num
    if features:        entry["features"] = features
    entry["pos_tags"] = sudachi_pos_tags
    return entry

def extract_entities(sentence: str):
    tokens_list  = tok.tokenize(sentence, tokenizer.Tokenizer.SplitMode.C)
    poly = score_politeness(sentence)
    raw_entities = []
    buffer = []
    pos_buffer = []

    def flush_buffer(role="predicate"):
        if buffer:
            result = create_entity_dict(
                "".join(buffer), role,
                [tag for tags in pos_buffer for tag in tags],
            )
            if result: raw_entities.append(result)
            buffer.clear()
            pos_buffer.clear()

    for tok_idx, m in enumerate(tokens_list):
        surface = m.surface()
        pos = m.part_of_speech()
        broad = pos[0]

        if surface == "の":
            no_type = classify_no(tokens_list, tok_idx)
            if no_type == "possessive":
                buffer.append(surface); pos_buffer.append(list(pos))
            else:
                flush_buffer("predicate")
            continue

        if surface in CASE_PARTICLE_ROLE or surface in {"。", "？", "?", "、", "！", "!"}:
            flush_buffer(CASE_PARTICLE_ROLE.get(surface, "predicate"))
        elif broad == "助動詞" and CONDITIONAL_TOPIC_RE.search("".join(buffer) + surface):
            flush_buffer("topic")
        elif broad == "動詞":
            flush_buffer("predicate")
            raw_entities.append({
                "original_entity": surface,
                "entity": m.dictionary_form(),
                "role": "action", "conf": 1.0,
            })
        elif broad in {"名詞", "代名詞", "固有名詞", "接尾辞", "数詞", "副詞", "接続詞"}:
            if should_flush_before(m, buffer) and buffer: flush_buffer("predicate")
            buffer.append(surface); pos_buffer.append(list(pos))

    flush_buffer("predicate")

    merged, i = [], 0
    while i < len(raw_entities):
        e = raw_entities[i]
        if e["role"] == "action":
            mv, aux, j = e["entity"], [], i + 1
            while j < len(raw_entities) and raw_entities[j]["role"] == "action":
                aux.append(raw_entities[j]["entity"]); j += 1

            if mv in LIGHT_VERBS and merged:
                prev = merged[-1]
                if prev["role"] == "predicate":
                    c = prev["entity"] + mv
                    prev.update({
                        "entity": c, "main_verb": c, "aux_verbs": aux,
                        "display": c + (("+" + "+".join(aux)) if aux else ""),
                        "role": "action",
                        "original_entity": prev.get("original_entity", "") + mv,
                    })
                    i = j; continue

            if mv in {"する", "いたす"} and merged:
                prev = merged[-1]
                if prev["role"] == "predicate":
                    mv = prev["entity"] + mv; merged.pop()

            if mv in {"つける", "付ける"} and merged:
                if merged[-1]["entity"] == "気":
                    mv = "気をつける"; merged.pop()

            display = mv + (("+" + "+".join(aux)) if aux else "")
            merged.append({
                "original_entity": e["original_entity"],
                "entity": mv, "display": display,
                "main_verb": mv, "aux_verbs": aux,
                "role": "action", "conf": 1.0,
            })
            i = j
        else:
            merged.append(e); i += 1

    current_verb = next((e for e in reversed(merged) if e["role"] == "action"), None)
    return merged, poly, current_verb

def extract_vocative_name_address_pairs(sentence: str):
    tokens = tok.tokenize(sentence, tokenizer.Tokenizer.SplitMode.A)
    pairs  = []
    for i, m in enumerate(tokens):
        if "人名" in m.part_of_speech():
            name, suf, ni = m.surface(), "", i + 1
            if ni < len(tokens) and tokens[ni].surface() in ADDRESS_SUFFIXES:
                suf = tokens[ni].surface(); ni += 1
            else:
                for s in ADDRESS_SUFFIXES:
                    if name.endswith(s) and len(name) > len(s):
                        suf, name = s, name[:-len(s)]; break
            if suf and (ni >= len(tokens)
                        or tokens[ni].surface() in {"、", "！", "!", "？", "?"}):
                pairs.append((name, suf))
    return pairs

def _most_recent_other_speaker(current_turn_idx: int, all_turns: list, current_sid: str):
    for turn in reversed(all_turns[:current_turn_idx]):
        if turn["speaker"] != current_sid:
            return turn["speaker"]
    return None

_DIST_THRESHOLD = 0.70

def _salience_key(item: dict, stack: list) -> float:
    rs = ROLE_SALIENCE.get(item["role"], 0.1)
    depth = next(
        (i for i, s in enumerate(reversed(stack)) if s["entity"] == item["entity"]),
        len(stack),
    )
    return rs * (RECENCY_DECAY ** depth)

def serialize_memory_state(mem: "DialogueMemory", registry: dict,
                            current_sid: str, listener_sids: set,
                            turn_resolved_zeros: list,
                            turn_resolved_prons: list) -> str:
    parts = []
    all_labels = {r.label() for r in registry.values()}

    spk = registry[current_sid]
    gen_tag = {"male": "M", "female": "F"}.get(spk.gender, "?")
    pol_tag = {"formal": "frm", "neutral": "neu", "informal": "inf"}.get(spk.politeness, "inf")
    parts.append(f"SPK={spk.label()}({gen_tag},{pol_tag})")

    for lst_sid in sorted(listener_sids):
        lst     = registry[lst_sid]
        gen_tag = {"male": "M", "female": "F"}.get(lst.gender, "?")
        pol_tag = {"formal": "frm", "neutral": "neu", "informal": "inf"}.get(lst.politeness, "inf")
        parts.append(f"LST={lst.label()}({gen_tag},{pol_tag})")

    ctx_cands = [
        item for item in mem.stack
        if item["entity"] not in all_labels
        and item["entity"] not in {"[ZERO_SUBJECT]", "[ZERO_OBJECT]"}
        and item["role"] not in {"action", "predicate"}
    ]
    ctx_cands.sort(key=lambda item: _salience_key(item, mem.stack), reverse=True)

    for item in ctx_cands[:2]:
        role_abbr = {"subject": "subj", "topic": "top", "object": "obj"}.get(
            item["role"], item["role"][:3])
        num_tag  = ",plur" if item.get("number") == "plural" else ""
        conf_val = item.get("conf", 1.0)
        parts.append(f"CTX={item['entity']}({role_abbr}{num_tag},{conf_val:.2f})")

    for clause_num, orig, entity, conf in turn_resolved_prons:
        parts.append(f"C{clause_num}_PRON={{{orig}:{entity}({conf:.2f})}}")

    for clause_num, role_tag, entry in turn_resolved_zeros:
        tag = f"C{clause_num}_{role_tag}"
        conf_val = entry.get("conf", 0.5)
        if conf_val >= _DIST_THRESHOLD or not entry.get("distribution"):
            parts.append(f"{tag}={entry['entity']}({conf_val:.2f})")
        else:
            dist_str = "|".join(f"{k}:{v:.2f}" for k, v in entry["distribution"].items())
            parts.append(f"{tag}={{{dist_str}}}")

    if mem.last_active_subject and mem.last_active_subject not in all_labels:
        las_conf = next(
            (item.get("conf", 1.0) for item in reversed(mem.stack)
             if item["entity"] == mem.last_active_subject),
            1.0,
        )
        parts.append(f"LAS={mem.last_active_subject}({las_conf:.2f})")

    return "[MEM: " + " ".join(parts) + "] "


class RealTimeDialogueProcessor:
    def __init__(self):
        self.speaker_ids = []
        self.registry = {}
        self.mem = DialogueMemory(self.speaker_ids, self.registry)
        self.all_turns = []
        self.previous_sid = None
        self.was_previous_question = False
        self.actual_speakers_so_far = set()

    def process_turn(self, speaker_id: str, raw_sentence: str) -> str:
        sid = speaker_id.strip()
        raw_sentence = raw_sentence.strip()
        turn_idx = len(self.all_turns)

        self.all_turns.append({"speaker": sid, "utterance": raw_sentence})

        if sid not in self.registry:
            unclaimed = [
                g for g in list(self.registry.keys())
                if str(g).startswith("Guest_") and g not in self.actual_speakers_so_far
            ]
            if len(unclaimed) == 1:
                gid = unclaimed[0]
                self.registry[sid] = self.registry.pop(gid)
                self.registry[sid].sid = sid
                if gid in self.speaker_ids: self.speaker_ids.remove(gid)
                self.speaker_ids.append(sid)
                old_lbl = self.mem._sid_to_label.pop(gid, f"Speaker_{gid}")
                self.mem._sid_to_label[sid] = old_lbl
                for item in self.mem.stack:
                    if item["entity"] == old_lbl and item["role"] == "speaker_ref":
                        item["entity"] = self.registry[sid].label(); break
                if old_lbl in self.mem.entity_features:
                    self.mem.entity_features[self.registry[sid].label()] = self.mem.entity_features.pop(old_lbl)
            else:
                self.registry[sid] = SpeakerMemory(sid)
                self.speaker_ids.append(sid)
                self.mem._sid_to_label[sid] = f"Speaker_{sid}"

        self.actual_speakers_so_far.add(sid)

        cur_lbl = self.registry[sid].label()
        if not any(i["entity"] == cur_lbl and i["role"] == "speaker_ref" for i in self.mem.stack):
            self.mem.stack.append({"entity": cur_lbl, "role": "speaker_ref",
                              "features": {"animate", "human"}, "conf": 1.0})
            self.mem.entity_features[cur_lbl] = {"animate", "human"}

        addressed_sids   = set()
        vocative_targets = {}

        for name, suf in extract_vocative_name_address_pairs(raw_sentence):
            target_sid = None
            for spk_id, spk_mem in self.registry.items():
                if spk_id != sid and (name == spk_mem.name or name in spk_mem.aliases):
                    target_sid = spk_id; break

            if not target_sid:
                recent = _most_recent_other_speaker(turn_idx, self.all_turns, sid)
                if recent and recent in self.registry and recent.startswith("Guest_") and self.registry[recent].name is None:
                    target_sid = recent

            if not target_sid:
                gid = f"Guest_{name}"
                if gid not in self.registry:
                    self.registry[gid] = SpeakerMemory(gid)
                    self.speaker_ids.append(gid)
                    self.mem._sid_to_label[gid] = f"Speaker_{gid}"
                    self.mem.stack.append({"entity": f"Speaker_{gid}", "role": "speaker_ref",
                                      "features": {"animate", "human"}, "conf": 1.0})
                    self.mem.entity_features[f"Speaker_{gid}"] = {"animate", "human"}
                target_sid = gid

            if target_sid:
                vocative_targets[name] = target_sid
                addressed_sids.add(target_sid)

        if not addressed_sids:
            recent = _most_recent_other_speaker(turn_idx, self.all_turns, sid)
            if recent and recent in self.registry:
                addressed_sids.add(recent)
            else:
                addressed_sids.update(s for s in self.speaker_ids if s != sid and s in self.registry)

        for a_sid in addressed_sids:
            a_lbl = self.registry[a_sid].label()
            if not any(i["entity"] == a_lbl and i["role"] == "speaker_ref" for i in self.mem.stack):
                self.mem.stack.append({"entity": a_lbl, "role": "speaker_ref",
                                  "features": {"animate", "human"}, "conf": 1.0})
                self.mem.entity_features[a_lbl] = {"animate", "human"}

        if self.previous_sid and self.previous_sid != sid and not self.was_previous_question:
            self.mem.reset_continuity()
        self.previous_sid = sid

        clauses = [
            c.strip()
            for c in re.split(
                r"(?<=[。？！?!])|(?<=て、)|(?<=たら、)|(?<=から、)|(?<=ので、)|(?<=が、)|(?<=けど、)",
                raw_sentence,
            )
            if c.strip()
        ] or [raw_sentence.strip()]

        turn_resolved_zeros = []
        turn_resolved_prons = []

        for clause_idx, sentence in enumerate(clauses):
            is_final = (clause_idx == len(clauses) - 1)
            is_question = bool(re.search(r"(か|？|\?)[^\w]*$", sentence))
            is_perm = bool(re.search(r"([てで]いい|[てで]もいい|なくていい)[^\w]*$", sentence))
            explicit_rq = bool(re.search(
                r"([てで]ください|ください|なさい|[てで]くれ|[てで]くれない|[てで]もらえない|[てで]ほしい)[^\w]*$",
                sentence))
            naked_te = is_final and bool(re.search(r"[てで][^\w]*$", sentence))
            is_imp = explicit_rq or naked_te

            clean = re.sub(r"[^\w\s]", "", sentence).strip()
            changed = True
            while changed:
                changed = False
                for filler in _DISCOURSE_FILLERS:
                    if clean.startswith(filler):
                        clean = clean[len(filler):].strip(); changed = True; break

            is_pure_greeting = (
                clean in _GREETING_PREDICATES or
                clean == "" or
                bool(re.search(r"(世話にな|世話様|よろしく)", clean))
            )

            ents, poly, cur_verb = extract_entities(sentence)
            update_politeness(self.registry[sid], poly)

            morph = parse_verb_morphology(sentence)
            morph["is_volitional"]    = morph["is_volitional"] or bool(
                re.search(r"(ましょう|ませんか|ようか)(?:か|？|\?|。)*$", sentence))
            morph["entails_listener"] = morph["entails_listener"] or is_perm or is_imp

            for name, suf in extract_vocative_name_address_pairs(sentence):
                ts = vocative_targets.get(name)
                if ts:
                    gen = gender_resolver.get_gender(name + suf)
                    self.mem.update_speaker_label(ts, name, gen, title=suf)

            cur_lbl = self.registry[sid].label()
            listener_labels = {self.registry[a].label() for a in addressed_sids}

            for e in ents:
                orig = e.get("original_entity", "")
                if orig in FIRST_PERSON_PRONOUNS:
                    if e.get("gender", "none") in {"male", "female"}:
                        self.registry[sid].gender = e["gender"]
                        for item in self.mem.stack:
                            if item["entity"] == cur_lbl: item["gender"] = e["gender"]
                    e["entity"] = cur_lbl; e["conf"] = 1.0
                elif orig in SECOND_PERSON_PRONOUNS:
                    for a in addressed_sids:
                        if e.get("gender", "none") in {"male", "female"}:
                            self.registry[a].gender = e["gender"]
                            for item in self.mem.stack:
                                if item["entity"] == self.registry[a].label():
                                    item["gender"] = e["gender"]
                    e["entity"] = list(listener_labels)[0] if listener_labels else "[ZERO_SUBJECT]"
                    e["conf"] = 0.95
                elif orig in EXPLICIT_PRONOUNS_TO_RESOLVE:
                    result, conf = self.mem.resolve_explicit_pronoun(e, cur_lbl, listener_labels)
                    if result:
                        e["entity"] = result["entity"]; e["conf"] = conf
                        turn_resolved_prons.append((clause_idx + 1, orig, result["entity"], conf))

            has_subj = any(e["role"] in {"subject", "topic"} for e in ents)
            has_obj  = any(e["role"] == "object" for e in ents)
            zsub = zobj = None

            if (morph["main_verb_dict"] or cur_verb) and not is_pure_greeting:
                if not has_subj:
                    res, bd, conf = self.mem.resolve(
                        "subject", morph, exclude_entities=[],
                        current_speaker_label=cur_lbl,
                        listener_labels=listener_labels,
                        is_question=is_question,
                        is_imperative=is_imp,
                        is_permission=is_perm,
                    )
                    if res:
                        ed = {"original_entity": "[ZERO_SUBJECT]", "entity": res["entity"],
                              "role": "subject", "conf": conf,
                              "distribution": get_top_k_distribution(bd)}
                        if res.get("gender", "none") != "none": ed["gender"] = res["gender"]
                        if res.get("number", "singular") == "plural": ed["number"] = res["number"]
                    else:
                        tgt = list(listener_labels)[0] if listener_labels else "Unknown_Listener"
                        ed  = {"original_entity": "[ZERO_SUBJECT]", "entity": tgt,
                               "role": "subject", "conf": 0.0}
                    zsub = ed; ents.append(zsub)
                    turn_resolved_zeros.append((clause_idx + 1, "ZSUB", zsub))

                if not has_obj:
                    if object_drop_warranted(morph):
                        sx = zsub["entity"] if zsub else next(
                            (e["entity"] for e in ents if e["role"] in {"subject", "topic"}), None)
                        res, bd, conf = self.mem.resolve(
                            "object", morph, exclude_entities=[sx] if sx else [],
                            current_speaker_label="", listener_labels=listener_labels,
                        )
                        if res:
                            zobj = {"original_entity": "[ZERO_OBJECT]", "entity": res["entity"],
                                    "role": "object", "conf": conf,
                                    "distribution": get_top_k_distribution(bd)}
                            if res.get("gender", "none") != "none": zobj["gender"] = res["gender"]
                            if res.get("number", "singular") == "plural": zobj["number"] = res["number"]
                            ents.append(zobj)
                            turn_resolved_zeros.append((clause_idx + 1, "ZOBJ", zobj))

            self.mem.update(ents)

            cs = next((e["entity"] for e in ents if e["role"] in {"subject", "topic"}), None)
            if cs:                               self.mem.last_active_subject = cs
            elif zsub:                           self.mem.last_active_subject = zsub["entity"]
            elif not (morph["main_verb_dict"] or cur_verb):
                for e in ents:
                    if "human" in e.get("features", set()) or e["role"] == "topic":
                        self.mem.last_active_subject = e["entity"]; break

            co = next((e["entity"] for e in ents if e["role"] == "object"), None)
            if co:        self.mem.last_active_object = co
            elif zobj:    self.mem.last_active_object = zobj["entity"]
        mem_prefix = serialize_memory_state(
            self.mem, self.registry, sid, addressed_sids,
            turn_resolved_zeros, turn_resolved_prons,
        )
        self.mem.stack = [item for item in self.mem.stack if item["role"] != "action"]
        self.was_previous_question = any(
            bool(re.search(r"(か|？|\?)[^\w]*$", c)) for c in clauses
        )
        return f"{sid}: {mem_prefix}{raw_sentence}"