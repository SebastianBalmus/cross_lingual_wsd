# Word Sense Disambiguation (WSD) for English and Romanian via definition selection
import os
import re
import json
import argparse
import random
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from tqdm import tqdm
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
    """Extract one label letter from generated text."""
    if not isinstance(text, str):
        return None
    t = text.strip()

    m = LETTER_RE.search(t)
    if m:
        ch = m.group(1)
        if ch in valid_letters:
            return ch

    # fallback: scan characters
    for ch in t:
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


def build_prompt(lemma: str, pos: str, sentence: str, senses: List[dict]) -> str:
    senses_block = "\n".join([f"{s['letter']}. {s['gloss']}" for s in senses])
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Target word: {lemma}\n"
        f"Part of speech: {pos}\n"
        f"Sentence: {sentence}\n\n"
        f"Possible senses:\n{senses_block}\n\n"
        f"Which sense (letter) matches the target word usage in the sentence?"
    )


def build_sense_inventory(
    df: pd.DataFrame,
    lemma_col: str,
    pos_col: str,
    sense_key_col: str,
    gloss_col: str,
) -> Dict[Tuple[str, str], List[dict]]:
    """
    Build inventory: (lemma,pos) -> [{sense_key, gloss, letter}, ...]
    Senses are deduped by sense_key and sorted for stable assignment (when not shuffling).
    """
    inv = {}
    grouped = df.groupby([lemma_col, pos_col], sort=False)

    for (lemma, pos), g in grouped:
        senses = (
            g[[sense_key_col, gloss_col]]
            .drop_duplicates(subset=[sense_key_col])
            .sort_values(sense_key_col, kind="stable")
            .reset_index(drop=True)
        )

        items = []
        for i, r in senses.iterrows():
            letter = chr(ord("A") + i)
            items.append(
                {
                    "sense_key": r[sense_key_col],
                    "gloss": r[gloss_col],
                    "letter": letter,  # stable ordering (only used if shuffle_senses=False)
                }
            )
        inv[(lemma, pos)] = items

    return inv


def build_gold_letter_map(
    sense_inv: Dict[Tuple[str, str], List[dict]],
) -> Dict[Tuple[str, str, str], str]:
    """Map (lemma,pos,sense_key) -> gold letter (stable ordering)."""
    gold = {}
    for (lemma, pos), senses in sense_inv.items():
        for s in senses:
            gold[(lemma, pos, s["sense_key"])] = s["letter"]
    return gold


def build_eval_rows(
    df: pd.DataFrame,
    sense_inv: Dict[Tuple[str, str], List[dict]],
    gold_letter: Dict[Tuple[str, str, str], str],
    lemma_col: str,
    pos_col: str,
    sense_key_col: str,
    sentence_col: str,
) -> List[dict]:
    """
    Each eval row contains: row_id, lemma, pos, sentence, sense_key, gold, senses(list).
    Skip groups with <2 senses.
    """
    rows = []
    for i, r in df.iterrows():
        lemma = r[lemma_col]
        pos = r[pos_col]
        sense_key = r[sense_key_col]
        senses = sense_inv.get((lemma, pos), [])
        if len(senses) < 2:
            continue

        gold = gold_letter.get((lemma, pos, sense_key))
        if gold is None:
            continue

        rows.append(
            {
                "row_id": i,
                "lemma": lemma,
                "pos": pos,
                "sense_key": sense_key,
                "sentence": r[sentence_col],
                "gold": gold,  # stable gold (used only if shuffle_senses=False)
                "senses": senses,  # list of dicts including sense_key/gloss/letter
                "no_choices": len(senses),
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

    senses_shuf = [dict(s) for s in senses]  # shallow copy each dict
    rng.shuffle(senses_shuf)

    gold_letter = None
    for i, s in enumerate(senses_shuf):
        letter = chr(ord("A") + i)
        s["letter"] = letter
        if s["sense_key"] == sense_key_gold:
            gold_letter = letter

    return senses_shuf, gold_letter


def evaluate_wsd(
    eval_rows: List[dict],
    gen,
    gen_kwargs: dict,
    baseline: Optional[str] = None,
    embedder=None,
    seed: int = 0,
    shuffle_senses: bool = False,
) -> Tuple[float, pd.DataFrame]:
    """
    shuffle_senses: if True, shuffle sense options per example and relabel A,B,C...
    """
    n = 0
    correct = 0
    outputs = []

    for ex in tqdm(eval_rows, desc="WSD eval"):
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
            raw = pred
        elif baseline == "lesk":
            pred = predict_lesk(ex["sentence"], senses_used)
            raw = pred
        elif baseline == "gloss_similarity":
            pred = predict_gloss_similarity(ex["sentence"], senses_used, embedder)
            raw = pred
        else:
            prompt = build_prompt(ex["lemma"], ex["pos"], ex["sentence"], senses_used)
            out = gen(prompt, **gen_kwargs)
            raw = (
                out[0]["generated_text"] if isinstance(out, list) and out else str(out)
            )
            pred = extract_letter(raw, valid_set)

        is_correct = pred == gold_used
        n += 1
        correct += int(is_correct)

        outputs.append(
            {
                "row_id": ex["row_id"],
                "lemma": ex["lemma"],
                "pos": ex["pos"],
                "sense_key": ex["sense_key"],
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
        "--seed", type=int, default=0, help="Random seed (also used for shuffling)"
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

    summary = {
        "model": args.model,
        "baseline": baseline,
        "random_baseline": random_baseline,
        "embedding_model": args.embedding_model if baseline == "gloss_similarity" else None,
        "dataset_path": args.dataset_path,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "shuffle_senses": bool(args.shuffle_senses),
    }

    # ========= ENGLISH =========
    en_inv = build_sense_inventory(
        df,
        lemma_col="en_lemma",
        pos_col="pos_en",
        sense_key_col="en_synset_name",
        gloss_col="en_gloss",
    )
    en_gold = build_gold_letter_map(en_inv)
    en_eval_rows = build_eval_rows(
        df,
        sense_inv=en_inv,
        gold_letter=en_gold,
        lemma_col="en_lemma",
        pos_col="pos_en",
        sense_key_col="en_synset_name",
        sentence_col="en_sentence",
    )

    en_acc, en_pred = evaluate_wsd(
        en_eval_rows,
        gen,
        GEN_KWARGS,
        baseline=baseline,
        embedder=embedder,
        seed=args.seed,
        shuffle_senses=args.shuffle_senses,
    )

    en_pred_filename = "wsd_en_predictions.csv"
    ro_pred_filename = "wsd_ro_predictions.csv"

    en_pred_path = os.path.join(args.results_dir, en_pred_filename)
    en_pred.to_csv(en_pred_path, index=False)
    summary["wsd_en"] = {
        "accuracy": en_acc,
        "n_eval": len(en_eval_rows),
        "predictions": en_pred_path,
    }
    print(f"[EN] WSD accuracy: {en_acc:.4f}  (n={len(en_eval_rows)})")
    print(f"[EN] Saved: {en_pred_path}")

    # ========= ROMANIAN =========
    ro_inv = build_sense_inventory(
        df,
        lemma_col="ro_lemma",
        pos_col="pos_ro",
        sense_key_col="ro_synset_id",
        gloss_col="ro_gloss",
    )
    ro_gold = build_gold_letter_map(ro_inv)
    ro_eval_rows = build_eval_rows(
        df,
        sense_inv=ro_inv,
        gold_letter=ro_gold,
        lemma_col="ro_lemma",
        pos_col="pos_ro",
        sense_key_col="ro_synset_id",
        sentence_col="ro_sentence",
    )

    ro_acc, ro_pred = evaluate_wsd(
        ro_eval_rows,
        gen,
        GEN_KWARGS,
        baseline=baseline,
        embedder=embedder,
        seed=args.seed,
        shuffle_senses=args.shuffle_senses,
    )
    ro_pred_path = os.path.join(args.results_dir, ro_pred_filename)
    ro_pred.to_csv(ro_pred_path, index=False)
    summary["wsd_ro"] = {
        "accuracy": ro_acc,
        "n_eval": len(ro_eval_rows),
        "predictions": ro_pred_path,
    }
    print(f"[RO] WSD accuracy: {ro_acc:.4f}  (n={len(ro_eval_rows)})")
    print(f"[RO] Saved: {ro_pred_path}")

    # --- save summary json ---
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
