from __future__ import annotations
import os, argparse, json, logging, random
from typing import List
import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger('robustness')

# --- Perturbations ---
KEYBOARD_NEIGHBORS = {
    'a':'qws', 'b':'vghn', 'c':'xdfv', 'd':'erfcxs', 'e':'rdsw', 'f':'rtdgcv', 'g':'tyfhvb', 'h':'yugjnb',
    'i':'uojk', 'j':'uikhmn', 'k':'ijolm', 'l':'kop', 'm':'njk', 'n':'bhjm', 'o':'ipl', 'p':'ol', 'q':'wa',
    'r':'etdf', 's':'wedxz', 't':'ryfg', 'u':'yihj', 'v':'cfgb', 'w':'qeas', 'x':'zsdc', 'y':'tugh', 'z':'asx'
}

def typo_swap(word: str) -> str:
    if len(word) < 2: return word
    i = random.randint(0, len(word)-2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]

def typo_keyboard(word: str) -> str:
    if not word: return word
    i = random.randint(0, len(word)-1)
    c = word[i].lower()
    if c in KEYBOARD_NEIGHBORS and KEYBOARD_NEIGHBORS[c]:
        rep = random.choice(KEYBOARD_NEIGHBORS[c])
        return word[:i] + rep + word[i+1:]
    return word

import regex as re
RE_PUNC = re.compile(r"[\p{P}\p{S}]")
RE_MULTI_WS = re.compile(r"\s+")

try:
    import regex as _regex  # better unicode categories
    RE_PUNC = _regex.compile(r"[\p{P}\p{S}]")
    RE_MULTI_WS = _regex.compile(r"\s+")
except Exception:
    pass


def perturb_text(s: str, mode: str) -> str:
    if mode == 'lower':
        return s.lower()
    if mode == 'upper':
        return s.upper()
    if mode == 'no_punc':
        return RE_PUNC.sub(' ', s)
    if mode == 'extra_ws':
        return RE_MULTI_WS.sub('   ', s)
    if mode == 'typo_swap':
        return ' '.join(typo_swap(w) for w in s.split())
    if mode == 'typo_keyboard':
        return ' '.join(typo_keyboard(w) for w in s.split())
    return s

PERTURBATIONS = ['lower','upper','no_punc','extra_ws','typo_swap','typo_keyboard']

# --- Evaluation ---

def predict(pipe, texts: List[str]):
    proba = None
    if hasattr(pipe, 'predict_proba'):
        proba = pipe.predict_proba(texts)[:,1]
    preds = pipe.predict(texts)
    return preds, proba


def run_robustness(model_path: str, data_csv: str, text_col: str, label_col: str, sample: int, out_csv: str, out_json: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pipe = joblib.load(model_path)

    df = pd.read_csv(data_csv)
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)

    base_preds, base_proba = predict(pipe, df[text_col].tolist())
    df_out = []
    for mode in PERTURBATIONS:
        texts_p = [perturb_text(t, mode) for t in df[text_col].tolist()]
        p_preds, p_proba = predict(pipe, texts_p)
        # flip rate: fraction where pred changed
        flips = (p_preds != base_preds).mean()
        # average confidence drop (only where proba available)
        conf_drop = None
        if base_proba is not None and p_proba is not None:
            conf_drop = float(np.mean(np.abs(p_proba - base_proba)))
        df_mode = pd.DataFrame({
            'orig_text': df[text_col],
            'perturb': mode,
            'perturbed_text': texts_p,
            'y_true': df[label_col],
            'y_pred_base': base_preds,
            'y_pred_perturbed': p_preds,
        })
        df_out.append((mode, flips, conf_drop, df_mode))
        logging.info("%s: flip_rate=%.4f, conf_drop=%s", mode, flips, None if conf_drop is None else f"{conf_drop:.4f}")

    # Save detailed rows
    details = pd.concat([d[3] for d in df_out], ignore_index=True)
    details.to_csv(out_csv, index=False)

    # Save summary
    summary = {
        'n_examples': int(len(df)),
        'perturbations': [
            {'name': mode, 'flip_rate': float(flips), 'avg_conf_change': (None if conf is None else float(conf))}
            for (mode, flips, conf, _) in df_out
        ]
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return out_csv, out_json


def main():
    ap = argparse.ArgumentParser(description='Robustness checks for text classifiers (simple perturbations)')
    ap.add_argument('--model', required=True, help='Path to models/trained_model.pkl')
    ap.add_argument('--data', required=True, help='CSV with texts/labels (e.g., data/processed/test.csv)')
    ap.add_argument('--text_col', default='review')
    ap.add_argument('--label_col', default='sentiment')
    ap.add_argument('--sample', type=int, default=800)
    ap.add_argument('--out_csv', default='reports/robustness_details.csv')
    ap.add_argument('--out_json', default='reports/robustness_summary.json')
    args = ap.parse_args()

    out_csv, out_json = run_robustness(args.model, args.data, args.text_col, args.label_col, args.sample, args.out_csv, args.out_json)
    LOGGER.info('Saved robustness details -> %s', out_csv)
    LOGGER.info('Saved robustness summary -> %s', out_json)


if __name__ == '__main__':
    main()