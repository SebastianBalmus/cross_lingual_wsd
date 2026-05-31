# Cross-lingual Word Sense Disambiguation (EN->RO, RO->EN) via definition selection

import os
import re
import json
import argparse
import random
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

import torch
from transformers import pipeline


SYSTEM_PROMPT = (
    "You are a careful linguist. Choose the correct sense label for the target word in the sentence.\n"
    "Answer with ONLY the single letter (A, B, C, ...). No other text.\n"
)

LETTER_RE = re.compile(r"\b([A-Z])\b")

GLOSS_SIMILARITY_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "her", "his", "in", "is", "it", "its", "of",
    "on", "or", "she", "that", "the", "their", "to", "was", "were", "with",
}

RO_STOPWORDS = {
    "a", "ai", "ale", "al", "am", "are", "au", "ca", "care", "ce", "cu",
    "de", "din", "dupa", "după", "este", "fi", "fie", "in", "în", "la",
    "le", "lui", "mai", "o", "pe", "pentru", "prin", "sau", "se", "si",
    "și", "sunt", "un", "unei", "unui",
}

STOPWORDS = EN_STOPWORDS | RO_STOPWORDS
TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def extract_letter(text: str, valid_letters: set) -> Optional[str]:
    text = (text or "").strip()
    m = LETTER_RE.search(text)
    if m:
        ch = m.group(1)
        if ch in valid_letters:
            return ch
    for ch in text:
        if ch in valid_letters:
            return ch
    return None




def normalize_baseline_name(model_name: str) -> Optional[str]:
    name = model_name.strip().lower().replace("-", "_")
    aliases = {
        "random": "random",
        "lesk": "lesk",
        "lesk_overlap": "lesk",
        "lexical_overlap": "lesk",
        "gloss_similarity": "gloss_similarity",
        "gloss_sim": "gloss_similarity",
        "multilingual_gloss_similarity": "gloss_similarity",
    }
    return aliases.get(name)


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The gloss_similarity baseline requires sentence-transformers. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    return SentenceTransformer(model_name)


def content_tokens(text: str) -> set:
    tokens = TOKEN_RE.findall(str(text).lower())
    return {tok for tok in tokens if tok not in STOPWORDS}


def predict_lesk(sentence: str, senses: List[dict]) -> str:
    sentence_tokens = content_tokens(sentence)
    best_letter = senses[0]["letter"]
    best_overlap = -1

    for sense in senses:
        overlap = len(sentence_tokens & content_tokens(sense["gloss"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_letter = sense["letter"]

    return best_letter


def predict_gloss_similarity(sentence: str, senses: List[dict], embedder) -> str:
    texts = [str(sentence)] + [str(s["gloss"]) for s in senses]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sentence_embedding = embeddings[0]
    gloss_embeddings = embeddings[1:]
    scores = gloss_embeddings @ sentence_embedding
    best_idx = max(range(len(senses)), key=lambda i: float(scores[i]))
    return senses[best_idx]["letter"]


def build_prompt_en_to_ro(
    en_lemma: str, pos_en: str, en_sentence: str, senses: List[dict]
) -> str:
    senses_block = "\n".join([f"{s['letter']}. {s['gloss']}" for s in senses])
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Target word (English): {en_lemma}\n"
        f"Part of speech: {pos_en}\n"
        f"Sentence (English): {en_sentence}\n\n"
        f"Possible senses (Romanian definitions):\n{senses_block}\n\n"
        f"Which sense (letter) matches the target word usage in the sentence?"
    )


def build_prompt_ro_to_en(
    ro_lemma: str, pos_ro: str, ro_sentence: str, senses: List[dict]
) -> str:
    senses_block = "\n".join([f"{s['letter']}. {s['gloss']}" for s in senses])
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Target word (Romanian): {ro_lemma}\n"
        f"Part of speech: {pos_ro}\n"
        f"Sentence (Romanian): {ro_sentence}\n\n"
        f"Possible senses (English definitions):\n{senses_block}\n\n"
        f"Which sense (letter) matches the target word usage in the sentence?"
    )


# ---------- Inventories ----------
def build_inventory_en_to_ro(
    df: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], List[dict]], Dict[Tuple[str, str, str], str]]:
    """
    (en_lemma, pos_en) -> senses with Romanian gloss options
    sense_key = ro_synset_id
    gold_letter maps (en_lemma,pos_en,ro_synset_id) -> stable letter
    """
    inv = {}
    gold_letter = {}

    for (lemma, pos), g in df.groupby(["en_lemma", "pos_en"], sort=True):
        senses = (
            g[["ro_synset_id", "ro_gloss"]]
            .drop_duplicates(subset=["ro_synset_id"])
            .sort_values("ro_synset_id", kind="stable")
            .reset_index(drop=True)
        )
        items = []
        for i, r in senses.iterrows():
            letter = chr(ord("A") + i)
            sense_key = r["ro_synset_id"]
            items.append(
                {"sense_key": sense_key, "gloss": r["ro_gloss"], "letter": letter}
            )
            gold_letter[(lemma, pos, sense_key)] = letter
        inv[(lemma, pos)] = items

    return inv, gold_letter


def build_inventory_ro_to_en(
    df: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], List[dict]], Dict[Tuple[str, str, str], str]]:
    """
    (ro_lemma, pos_ro) -> senses with English gloss options
    sense_key = ro_synset_id
    gold_letter maps (ro_lemma,pos_ro,ro_synset_id) -> stable letter
    """
    inv = {}
    gold_letter = {}

    for (lemma, pos), g in df.groupby(["ro_lemma", "pos_ro"], sort=True):
        senses = (
            g[["ro_synset_id", "en_gloss"]]
            .drop_duplicates(subset=["ro_synset_id"])
            .sort_values("ro_synset_id", kind="stable")
            .reset_index(drop=True)
        )
        items = []
        for i, r in senses.iterrows():
            letter = chr(ord("A") + i)
            sense_key = r["ro_synset_id"]
            items.append(
                {"sense_key": sense_key, "gloss": r["en_gloss"], "letter": letter}
            )
            gold_letter[(lemma, pos, sense_key)] = letter
        inv[(lemma, pos)] = items

    return inv, gold_letter


# ---------- Eval row builders ----------
def make_eval_rows_en_to_ro(df: pd.DataFrame, inv, gold_letter) -> List[dict]:
    rows = []
    for i, r in df.iterrows():
        lemma = r["en_lemma"]
        pos = r["pos_en"]
        senses = inv.get((lemma, pos), [])
        if len(senses) < 2:
            continue

        sense_key = r["ro_synset_id"]
        gold = gold_letter.get((lemma, pos, sense_key))
        if gold is None:
            continue

        rows.append(
            {
                "row_id": i,
                "lemma": lemma,
                "pos": pos,
                "sentence": r["en_sentence"],
                "sense_key": sense_key,
                "gold": gold,  # stable (only used if shuffle_senses=False)
                "senses": senses,  # list of dicts
                "no_choices": len(senses),
                "direction": "en2ro",
            }
        )
    return rows


def make_eval_rows_ro_to_en(df: pd.DataFrame, inv, gold_letter) -> List[dict]:
    rows = []
    for i, r in df.iterrows():
        lemma = r["ro_lemma"]
        pos = r["pos_ro"]
        senses = inv.get((lemma, pos), [])
        if len(senses) < 2:
            continue

        sense_key = r["ro_synset_id"]
        gold = gold_letter.get((lemma, pos, sense_key))
        if gold is None:
            continue

        rows.append(
            {
                "row_id": i,
                "lemma": lemma,
                "pos": pos,
                "sentence": r["ro_sentence"],
                "sense_key": sense_key,
                "gold": gold,  # stable (only used if shuffle_senses=False)
                "senses": senses,  # list of dicts
                "no_choices": len(senses),
                "direction": "ro2en",
            }
        )
    return rows


def _shuffle_and_relabel_senses(
    senses: List[dict],
    sense_key_gold: str,
    seed: int,
) -> Tuple[List[dict], str]:
    """
    Deterministically shuffle senses and reassign labels A,B,C...
    Returns (shuffled_senses_with_new_letters, gold_letter_for_this_shuffle).
    """
    rng = random.Random(seed)
    senses_shuf = [dict(s) for s in senses]
    rng.shuffle(senses_shuf)

    gold_letter = None
    for i, s in enumerate(senses_shuf):
        letter = chr(ord("A") + i)
        s["letter"] = letter
        if s["sense_key"] == sense_key_gold:
            gold_letter = letter

    return senses_shuf, gold_letter


# ---------- Evaluators ----------
def eval_direction(
    eval_rows: List[dict],
    gen,
    gen_kwargs: dict,
    baseline: Optional[str],
    embedder,
    seed: int,
    shuffle_senses: bool,
) -> Tuple[float, pd.DataFrame]:
    outputs = []
    correct = 0
    n = 0

    direction = eval_rows[0]["direction"] if eval_rows else ""

    for ex in tqdm(
        eval_rows, desc=f"Cross-lingual WSD {direction.upper() if eval_rows else ''}".strip()
    ):
        seed = seed + int(ex["row_id"])
        rng = random.Random(seed)

        senses = ex["senses"]

        if shuffle_senses:
            senses_used, gold_used = _shuffle_and_relabel_senses(
                senses=senses,
                sense_key_gold=ex["sense_key"],
                seed=seed,
            )
        else:
            senses_used = senses
            gold_used = ex["gold"]

        valid_letters = [s["letter"] for s in senses_used]
        valid_set = set(valid_letters)

        if baseline == "random":
            pred = rng.choice(valid_letters)
        elif baseline == "lesk":
            pred = predict_lesk(ex["sentence"], senses_used)
        elif baseline == "gloss_similarity":
            pred = predict_gloss_similarity(ex["sentence"], senses_used, embedder)
        else:
            if ex["direction"] == "en2ro":
                prompt = build_prompt_en_to_ro(
                    ex["lemma"], ex["pos"], ex["sentence"], senses_used
                )
            else:
                prompt = build_prompt_ro_to_en(
                    ex["lemma"], ex["pos"], ex["sentence"], senses_used
                )

            out = gen(prompt, **gen_kwargs)
            text = (
                out[0]["generated_text"] if isinstance(out, list) and out else str(out)
            )
            pred = extract_letter(text, valid_set)

        is_correct = pred == gold_used
        correct += int(is_correct)
        n += 1

        outputs.append(
            {
                "row_id": ex["row_id"],
                "lemma": ex["lemma"],
                "pos": ex["pos"],
                "ro_synset_id": ex["sense_key"],
                "no_choices": ex["no_choices"],
                "gold": gold_used,
                "pred": pred,
                "correct": bool(is_correct),
            }
        )

    acc = correct / n if n else 0.0
    return acc, pd.DataFrame(outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        required=True,
        help=(
            'HF model name/path (e.g. mistralai/Mistral-7B-Instruct-v0.3), '
            'or baseline name: "random", "lesk", "gloss_similarity"'
        ),
    )
    ap.add_argument(
        "-r", "--results_dir", required=True, help="Directory to save outputs"
    )
    ap.add_argument("-d", "--dataset_path", required=True, help="CSV path with dataset")
    ap.add_argument(
        "--max_rows", type=int, default=None, help="Optional cap for quick tests"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (also used for shuffling/baseline)",
    )
    ap.add_argument(
        "--shuffle_senses",
        action="store_true",
        help="Shuffle sense options per example and relabel A,B,C... (recommended control)",
    )
    ap.add_argument(
        "--embedding_model",
        default=GLOSS_SIMILARITY_MODEL,
        help="SentenceTransformer model used by the gloss_similarity baseline",
    )
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    df = pd.read_csv(args.dataset_path).reset_index(drop=True)
    if args.max_rows is not None:
        df = df.head(args.max_rows).reset_index(drop=True)

    needed = [
        "en_lemma",
        "pos_en",
        "en_gloss",
        "ro_lemma",
        "pos_ro",
        "ro_synset_id",
        "ro_gloss",
        "en_sentence",
        "ro_sentence",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    baseline = normalize_baseline_name(args.model)
    random_baseline = baseline == "random"

    gen = None
    embedder = None
    GEN_KWARGS = dict(
        max_new_tokens=5,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
    )

    if baseline == "gloss_similarity":
        embedder = load_sentence_transformer(args.embedding_model)
    elif baseline is None:
        gen = pipeline(
            "text-generation",
            model=args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    # ---------- EN -> RO ----------
    inv_en_ro, gold_en_ro = build_inventory_en_to_ro(df)
    eval_rows_en_ro = make_eval_rows_en_to_ro(df, inv_en_ro, gold_en_ro)

    acc_en_ro, df_pred_en_ro = eval_direction(
        eval_rows_en_ro,
        gen,
        GEN_KWARGS,
        baseline=baseline,
        embedder=embedder,
        seed=args.seed,
        shuffle_senses=args.shuffle_senses,
    )

    # ---------- RO -> EN ----------
    inv_ro_en, gold_ro_en = build_inventory_ro_to_en(df)
    eval_rows_ro_en = make_eval_rows_ro_to_en(df, inv_ro_en, gold_ro_en)

    acc_ro_en, df_pred_ro_en = eval_direction(
        eval_rows_ro_en,
        gen,
        GEN_KWARGS,
        baseline=baseline,
        embedder=embedder,
        seed=args.seed,
        shuffle_senses=args.shuffle_senses,
    )

    en2ro_name = "cross_lingual_wsd_en2ro_predictions.csv"
    ro2en_name = "cross_lingual_wsd_ro2en_predictions.csv"

    out_en2ro = os.path.join(args.results_dir, en2ro_name)
    out_ro2en = os.path.join(args.results_dir, ro2en_name)

    df_pred_en_ro.to_csv(out_en2ro, index=False)
    df_pred_ro_en.to_csv(out_ro2en, index=False)

    summary = {
        "task": "cross_lingual_wsd",
        "model": args.model,
        "baseline": baseline,
        "random_baseline": random_baseline,
        "embedding_model": args.embedding_model if baseline == "gloss_similarity" else None,
        "shuffle_senses": bool(args.shuffle_senses),
        "seed": args.seed,
        "dataset_path": args.dataset_path,
        "num_rows_dataset": int(len(df)),
        "en2ro": {
            "num_eval_rows": int(len(eval_rows_en_ro)),
            "top1_accuracy": float(acc_en_ro),
            "predictions": out_en2ro,
        },
        "ro2en": {
            "num_eval_rows": int(len(eval_rows_ro_en)),
            "top1_accuracy": float(acc_ro_en),
            "predictions": out_ro2en,
        },
    }

    with open(
        os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("EN->RO top-1 accuracy:", acc_en_ro, "on", len(eval_rows_en_ro), "rows")
    print("RO->EN top-1 accuracy:", acc_ro_en, "on", len(eval_rows_ro_en), "rows")
    print("Saved outputs to:", args.results_dir)


if __name__ == "__main__":
    main()
