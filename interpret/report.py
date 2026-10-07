# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Morfoloji Mikroskobu Raporları

Toplanan aktivasyonlardan sonda/SAE/karşılaştırma raporları üretir ve bunları
JSON + bağımsız (self-contained) HTML olarak yazar. HTML dış kaynak (CDN,
font, betik) yüklemez; CSS/JS/SVG satır içidir, açık/koyu tema
``prefers-color-scheme`` ile seçilir, dar ekranlara uyar.

Ana fonksiyonlar:
    run_probe_suite      — tüm özellikler × tüm katmanlar sondası (+ kontrol görevi)
    build_probe_report   — sonda raporu sözlüğü (token görünümü + MoE tablosu dahil)
    render_probe_html    — sonda raporu HTML'i
    build_sae_report / render_sae_html
    build_compare_report / render_compare_html — iki checkpoint farkı
    write_report         — out_dir/<ad>.json ve out_dir/<ad>.html yazar
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import torch

from interpret.activations import CollectedActivations
from interpret.features import FEATURES, compute_features
from interpret.probes import group_val_mask, probe_all_layers

REPORT_VERSION = "toprak-interpret-v1"
MORPH_CLASS_NAMES = ["kök", "ek", "özel"]


# ════════════════════════════════════════════════════════════
# Hesaplama
# ════════════════════════════════════════════════════════════

def texts_fingerprint(texts: Sequence[str]) -> str:
    """Metin kümesinin SHA-256 parmak izi (karşılaştırma uyumluluğu için)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def run_probe_suite(
    collected: CollectedActivations,
    features: Optional[Sequence[str]] = None,
    site: str = "resid_post",
    seeds: Sequence[int] = (0,),
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    val_split: float = 0.25,
    control: bool = True,
    min_labeled: int = 20,
):
    """
    Her özellik için tüm katmanlarda sonda eğit.

    Returns:
        (results, probes, labels, val_mask)
        results: {özellik: {"rows": [...], "num_labeled", "class_counts"}}
        probes:  {özellik: {katman: ProbeResult}} (ilk tohum)
        labels:  {özellik: (N,) etiketler}
        val_mask: (N,) ilk tohumun cümle düzeyindeki doğrulama maskesi
    """
    names = list(FEATURES) if features is None else list(features)
    labels = compute_features(collected.token_strings, collected.token_ids, names)
    layers = collected.layer_dict(site)
    groups = collected.sentence_index
    val_mask = group_val_mask(groups, val_split, seeds[0])
    results, probes = {}, {}
    for name in names:
        y = labels[name]
        valid = y >= 0
        n_classes = FEATURES[name].num_classes
        counts = torch.bincount(y[valid], minlength=n_classes).tolist()
        entry = {"num_labeled": int(valid.sum()), "class_counts": counts, "rows": [], "skipped": None}
        enough = (
            int(valid.sum()) >= min_labeled
            and sum(1 for c in counts if c > 0) >= 2
            and bool((valid & val_mask).any())
            and bool((valid & ~val_mask).any())
        )
        if not enough:
            entry["skipped"] = "yetersiz etiketli örnek veya tek sınıf"
            results[name] = entry
            continue
        out = probe_all_layers(
            layers, y, num_classes=n_classes, token_ids=collected.flat_token_ids,
            control=control, groups=groups, seeds=seeds,
            epochs=epochs, lr=lr, weight_decay=weight_decay, val_split=val_split,
        )
        entry["rows"] = out["rows"]
        results[name] = entry
        probes[name] = out["probes"]
    return results, probes, labels, val_mask


def moe_routing_report(collected: CollectedActivations, num_experts: Optional[int] = None) -> Optional[dict]:
    """MoE katmanları için uzman × morfolojik sınıf tablosu (yoksa None)."""
    if not collected.routing:
        return None
    from model.moe import routing_by_morph_class

    out = {}
    for layer in sorted(collected.routing):
        routing = collected.routing[layer]
        E = num_experts or int(routing.max().item()) + 1
        table = routing_by_morph_class(routing.unsqueeze(0), collected.morph_classes.unsqueeze(0), E)
        col_tot = table.sum(dim=0).clamp_min(1).float()
        out[f"L{layer + 1}"] = {
            "num_experts": E,
            "top_k": int(routing.size(-1)),
            "counts": table.tolist(),
            "share_of_class": (table.float() / col_tot).tolist(),
        }
    return {"classes": MORPH_CLASS_NAMES, "layers": out}


def _best_layer(rows: List[dict], key: str = "selectivity") -> Optional[str]:
    if not rows:
        return None
    k = key if key in rows[0] else "accuracy"
    return max(rows, key=lambda r: r[k])["layer"]


def build_probe_report(
    collected: CollectedActivations,
    features: Optional[Sequence[str]] = None,
    site: str = "resid_post",
    seeds: Sequence[int] = (0,),
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    val_split: float = 0.25,
    control: bool = True,
    view_sentences: int = 40,
    metadata: Optional[dict] = None,
    num_experts: Optional[int] = None,
) -> dict:
    """
    Tam sonda raporu: katman eğrileri, token görünümü verisi, MoE tablosu.

    Token görünümündeki olasılıklar ilk tohumun sondalarından gelir; eğitim
    cümlelerinde bunlar örneklem-içi (in-sample) değerlerdir — rapor her
    cümleyi "eğitim" / "doğrulama" diye işaretler.
    """
    results, probes, labels, val_mask = run_probe_suite(
        collected, features, site, seeds, epochs, lr, weight_decay, val_split, control
    )
    layer_acts = collected.layer_dict(site)
    layer_names = list(layer_acts)

    # Token görünümü
    n_view = min(view_sentences, len(collected.token_strings))
    view_rows = torch.isin(collected.sentence_index, torch.arange(n_view))
    sent_of_row = collected.sentence_index[view_rows]
    sentences = []
    for s in range(n_view):
        rows_s = (collected.sentence_index == s)
        sentences.append({
            "text": collected.texts[s] if s < len(collected.texts) else "",
            "tokens": collected.token_strings[s],
            "split": "doğrulama" if bool(val_mask[rows_s].any()) else "eğitim",
            "labels": {f: labels[f][rows_s].tolist() for f in results},
        })
    probs: Dict[str, Dict[str, List]] = {}
    for f, by_layer in probes.items():
        probs[f] = {}
        for layer, res in by_layer.items():
            p = res.predict_proba(layer_acts[layer][view_rows])
            pct = (p * 100).round().to(torch.int64)
            per_sent = [pct[sent_of_row == s].tolist() for s in range(n_view)]
            probs[f][layer] = per_sent

    feats_out = {}
    for name, entry in results.items():
        spec = FEATURES[name]
        feats_out[name] = {
            "classes": list(spec.classes),
            "description": spec.description,
            **entry,
            "best_layer": _best_layer(entry["rows"]),
        }

    meta = {
        "num_sentences": len(collected.token_strings),
        "num_tokens": collected.num_tokens,
        "site": site,
        "seeds": list(seeds),
        "epochs": epochs,
        "val_split": val_split,
        "texts_sha256": texts_fingerprint(collected.texts),
    }
    meta.update(metadata or {})
    return {
        "version": REPORT_VERSION,
        "kind": "probe",
        "metadata": meta,
        "layers": layer_names,
        "features": feats_out,
        "token_view": {"sentences": sentences, "probs": probs},
        "moe": moe_routing_report(collected, num_experts),
    }


def build_sae_report(
    train_result,
    collected: CollectedActivations,
    acts: torch.Tensor,
    layer_name: str,
    top_features: int = 24,
    contexts_per_feature: int = 8,
    associate: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    SAE raporu: eğitim özeti, en sık ateşlenen latentlerin en güçlü
    bağlamları ve her (özellik, sınıf) için en ilişkili latentler.
    """
    from interpret.sae import encode_all, feature_label_association, feature_top_tokens

    sae = train_result.sae
    z = encode_all(sae, acts)
    fires = (z > 0).sum(0)
    alive = torch.nonzero(fires > 0).flatten()
    order = alive[fires[alive].argsort(descending=True)][:top_features].tolist()
    flat_strs = collected.flat_token_strings()
    tops = feature_top_tokens(
        sae, acts, flat_strs, n=contexts_per_feature, features=order,
        sentence_ids=collected.sentence_index, latents=z,
    )
    names = list(associate) if associate is not None else list(FEATURES)
    labels = compute_features(collected.token_strings, collected.token_ids, names)
    assoc = {}
    extra_feats = set()
    for name in names:
        spec = FEATURES[name]
        assoc[name] = {}
        for c, cname in enumerate(spec.classes):
            ranked = feature_label_association(sae, acts, labels[name], positive_class=c, top=5, latents=z)
            if ranked:
                assoc[name][cname] = ranked
                extra_feats.update(r["feature"] for r in ranked[:2])
    extra = sorted(extra_feats - set(order))
    if extra:
        tops.update(feature_top_tokens(
            sae, acts, flat_strs, n=contexts_per_feature, features=extra,
            sentence_ids=collected.sentence_index, latents=z,
        ))
    meta = {
        "layer": layer_name,
        "num_tokens": collected.num_tokens,
        "texts_sha256": texts_fingerprint(collected.texts),
    }
    meta.update(metadata or {})
    return {
        "version": REPORT_VERSION,
        "kind": "sae",
        "metadata": meta,
        "training": train_result.summary(),
        "top_features": [
            {"feature": f, "fires": int(fires[f]), "contexts": tops.get(f, [])} for f in order
        ],
        "feature_contexts": {str(f): tops[f] for f in extra},
        "associations": assoc,
    }


def build_compare_report(report_a: dict, report_b: dict, name_a: str = "A", name_b: str = "B") -> dict:
    """
    İki sonda raporunun katman bazında farkı (B − A).

    Gürültü tahmini: doğrulama doğruluğunun binom standart hatası
    sqrt(p(1−p)/n) ve (varsa) tohumlar arası standart sapma birleştirilir;
    |Δ| > 2·SE olan hücreler "belirgin" işaretlenir. Bu kaba bir sezgiseldir —
    kesin sonuç için birden çok tohum ve bağımsız metin kümesi kullanın.
    """
    warnings = []
    ma, mb = report_a.get("metadata", {}), report_b.get("metadata", {})
    if ma.get("texts_sha256") != mb.get("texts_sha256"):
        warnings.append("İki rapor farklı metinlerle üretilmiş; farklar karşılaştırılamayabilir.")
    if ma.get("tokenizer_sha256") and ma.get("tokenizer_sha256") != mb.get("tokenizer_sha256"):
        warnings.append("Tokenizer'lar farklı.")
    if ma.get("model_config") and ma.get("model_config") != mb.get("model_config"):
        warnings.append("Model mimarileri farklı; ablation karşılaştırması aynı mimari gerektirir.")
    if report_a.get("layers") != report_b.get("layers"):
        warnings.append("Katman listeleri farklı; yalnız ortak katmanlar karşılaştırıldı.")
    layers = [l for l in report_a.get("layers", []) if l in report_b.get("layers", [])]
    feats = {}
    for name, fa in report_a.get("features", {}).items():
        fb = report_b.get("features", {}).get(name)
        if fb is None or not fa.get("rows") or not fb.get("rows"):
            continue
        ra = {r["layer"]: r for r in fa["rows"]}
        rb = {r["layer"]: r for r in fb["rows"]}
        rows = []
        for layer in layers:
            if layer not in ra or layer not in rb:
                continue
            a, b = ra[layer], rb[layer]

            def var(r):
                p, n = r["accuracy"], max(1, r.get("num_val", 1))
                return p * (1 - p) / n + r.get("accuracy_std", 0.0) ** 2

            se = math.sqrt(var(a) + var(b))
            delta = b["accuracy"] - a["accuracy"]
            row = {
                "layer": layer,
                "accuracy_a": a["accuracy"],
                "accuracy_b": b["accuracy"],
                "delta_accuracy": delta,
                "se": se,
                "notable": bool(se > 0 and abs(delta) > 2 * se),
            }
            if "selectivity" in a and "selectivity" in b:
                row["selectivity_a"] = a["selectivity"]
                row["selectivity_b"] = b["selectivity"]
                row["delta_selectivity"] = b["selectivity"] - a["selectivity"]
            rows.append(row)
        if rows:
            feats[name] = {
                "classes": fa.get("classes", []),
                "description": fa.get("description", ""),
                "rows": rows,
                "mean_delta_accuracy": sum(r["delta_accuracy"] for r in rows) / len(rows),
            }
    return {
        "version": REPORT_VERSION,
        "kind": "compare",
        "names": [name_a, name_b],
        "metadata": {"a": ma, "b": mb},
        "warnings": warnings,
        "layers": layers,
        "features": feats,
    }


def write_report(out_dir: str, stem: str, report: dict, html_text: str) -> dict:
    """``out_dir/<stem>.json`` ve ``out_dir/<stem>.html`` yaz; yolları döndür."""
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, f"{stem}.json")
    hpath = os.path.join(out_dir, f"{stem}.html")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    with open(hpath, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    return {"json": jpath, "html": hpath}


# ════════════════════════════════════════════════════════════
# HTML
# ════════════════════════════════════════════════════════════

_CSS = """
:root{color-scheme:light;--bg:#f6f6f4;--surface:#fcfcfb;--border:#e2e1dc;--text:#0b0b0b;
--text2:#52514e;--muted:#8a8984;--s1:#2a78d6;--s2:#eb6834;--neg:#e34948;--grid:#e8e7e2;--mid:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111110;
--surface:#1a1a19;--border:#33332f;--text:#fff;--text2:#c3c2b7;--muted:#8f8e86;--s1:#3987e5;--s2:#d95926;
--neg:#e66767;--grid:#2a2a27;--mid:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--border:#33332f;--text:#fff;
--text2:#c3c2b7;--muted:#8f8e86;--s1:#3987e5;--s2:#d95926;--neg:#e66767;--grid:#2a2a27;--mid:#383835}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.2rem;margin:32px 0 8px}h3{font-size:1rem;margin:0 0 4px}
p.lead{color:var(--text2);margin:0 0 16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px;color:var(--text2);font-size:.85rem}
.meta b{color:var(--text)}
.desc{color:var(--text2);font-size:.85rem;margin:0 0 8px}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:.8rem;color:var(--text2);margin:4px 0}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--text2);font-size:11px}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.82rem;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--text2);font-weight:600}
details summary{cursor:pointer;color:var(--text2);font-size:.85rem}
.controls{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:end;margin-bottom:12px}
.controls label{display:flex;flex-direction:column;font-size:.8rem;color:var(--text2);gap:2px}
select{font:inherit;padding:4px 6px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);max-width:100%}
.tokens{line-height:2.1;word-break:break-word}
.tok{display:inline-block;padding:0 3px;margin:1px 1px;border-radius:4px;border:1px solid transparent;cursor:default;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.9rem}
.tok.nolabel{border-style:dashed;border-color:var(--border)}
.tok.wrong{outline:2px solid var(--neg);outline-offset:-1px}
.badge{display:inline-block;font-size:.75rem;padding:1px 8px;border-radius:999px;border:1px solid var(--border);color:var(--text2)}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:6px 10px;font-size:.8rem;box-shadow:0 4px 16px rgba(0,0,0,.15);display:none;z-index:10;max-width:280px}
.ctx{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.82rem;white-space:pre-wrap;word-break:break-word}
.ctx mark{background:color-mix(in srgb,var(--s1) 35%,transparent);color:inherit;border-radius:3px;padding:0 2px}
.pos{color:var(--s1)}.negv{color:var(--neg)}
.warn{border-left:4px solid var(--s2);padding:8px 12px;background:var(--surface);margin:8px 0}
"""


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _json_script(obj, element_id: str) -> str:
    data = json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{data}</script>'


def _page(title: str, body: str, script: str = "") -> str:
    return (
        "<!doctype html>\n<html lang=\"tr\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><main>"
        f"{body}</main><div id=\"tip\"></div>"
        f"{('<script>' + script + '</script>') if script else ''}</body></html>\n"
    )


def _meta_block(meta: dict, keys: Sequence[str]) -> str:
    parts = [f"<span>{_esc(k)}: <b>{_esc(meta[k])}</b></span>" for k in keys if k in meta and meta[k] is not None]
    return f'<div class="meta">{"".join(parts)}</div>'


def layer_bar_svg(rows: List[dict], width: int = 560, height: int = 200) -> str:
    """Katman × doğruluk (mavi) ve kontrol doğruluğu (turuncu) çubukları; çoğunluk tabanı kesikli çizgi."""
    pad_l, pad_r, pad_t, pad_b = 34, 8, 8, 26
    n = max(1, len(rows))
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    slot = plot_w / n
    has_ctrl = any("control_accuracy" in r for r in rows)
    bar_w = max(2.0, min(18.0, (slot - 6) / (2 if has_ctrl else 1) - 2))
    y = lambda v: pad_t + plot_h * (1 - max(0.0, min(1.0, v)))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Katman başına sonda doğruluğu">']
    for g in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/>')
        out.append(f'<text x="{pad_l - 4}" y="{y(g) + 4:.1f}" text-anchor="end">{g:.2f}</text>')
    for i, r in enumerate(rows):
        x0 = pad_l + i * slot + (slot - bar_w * (2 if has_ctrl else 1) - (2 if has_ctrl else 0)) / 2
        acc = r["accuracy"]
        tip = f'{r["layer"]}: doğruluk {acc:.3f}'
        if "control_accuracy" in r:
            tip += f', kontrol {r["control_accuracy"]:.3f}, seçicilik {r["selectivity"]:+.3f}'
        tip += f', çoğunluk {r["majority_baseline"]:.3f}, makro-F1 {r["macro_f1"]:.3f}'
        h = y(0) - y(acc)
        out.append(f'<g><title>{_esc(tip)}</title>'
                   f'<rect x="{x0:.1f}" y="{y(acc):.1f}" width="{bar_w:.1f}" height="{max(h, 0):.1f}" rx="3" fill="var(--s1)"/>')
        if has_ctrl:
            c = r.get("control_accuracy", 0.0)
            out.append(f'<rect x="{x0 + bar_w + 2:.1f}" y="{y(c):.1f}" width="{bar_w:.1f}" height="{max(y(0) - y(c), 0):.1f}" rx="3" fill="var(--s2)"/>')
        out.append(f'<rect x="{pad_l + i * slot:.1f}" y="{pad_t}" width="{slot:.1f}" height="{plot_h}" fill="transparent"/></g>')
        mb = r["majority_baseline"]
        out.append(f'<line x1="{pad_l + i * slot + 2:.1f}" x2="{pad_l + (i + 1) * slot - 2:.1f}" y1="{y(mb):.1f}" y2="{y(mb):.1f}" '
                   f'stroke="var(--text2)" stroke-width="1.5" stroke-dasharray="3 3"/>')
        out.append(f'<text x="{pad_l + (i + .5) * slot:.1f}" y="{height - 8}" text-anchor="middle">{_esc(r["layer"])}</text>')
    out.append("</svg>")
    return "".join(out)


STROKE = ' stroke="var(--text)" stroke-width="1.5"'


def delta_bar_svg(rows: List[dict], key: str = "delta_accuracy", width: int = 560, height: int = 180) -> str:
    """Katman başına B − A farkı; mavi = artış, kırmızı = azalış, sıfır çizgisi ortada."""
    pad_l, pad_r, pad_t, pad_b = 40, 8, 8, 26
    n = max(1, len(rows))
    vals = [r.get(key, 0.0) for r in rows]
    lim = max(0.05, max((abs(v) for v in vals), default=0.05))
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    slot = plot_w / n
    bar_w = max(3.0, min(22.0, slot - 8))
    y = lambda v: pad_t + plot_h * (0.5 - v / (2 * lim))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Katman başına fark">']
    for g in (-lim, -lim / 2, 0, lim / 2, lim):
        out.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/>')
        out.append(f'<text x="{pad_l - 4}" y="{y(g) + 4:.1f}" text-anchor="end">{g:+.2f}</text>')
    out.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--muted)"/>')
    for i, r in enumerate(rows):
        v = r.get(key, 0.0)
        x0 = pad_l + i * slot + (slot - bar_w) / 2
        top, bot = (y(v), y(0)) if v >= 0 else (y(0), y(v))
        color = "var(--s1)" if v >= 0 else "var(--neg)"
        tip = f'{r["layer"]}: Δ {v:+.3f} (A {r["accuracy_a"]:.3f} → B {r["accuracy_b"]:.3f}, SE {r["se"]:.3f})'
        mark = " *" if r.get("notable") else ""
        out.append(f'<g><title>{_esc(tip + mark)}</title><rect x="{x0:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
                   f'height="{max(bot - top, 0.5):.1f}" rx="3" fill="{color}"'
                   f'{STROKE if r.get("notable") else ""}/>'
                   f'<rect x="{pad_l + i * slot:.1f}" y="{pad_t}" width="{slot:.1f}" height="{plot_h}" fill="transparent"/></g>')
        out.append(f'<text x="{pad_l + (i + .5) * slot:.1f}" y="{height - 8}" text-anchor="middle">{_esc(r["layer"])}{mark}</text>')
    out.append("</svg>")
    return "".join(out)


def _probe_table(rows: List[dict]) -> str:
    has_ctrl = rows and "control_accuracy" in rows[0]
    head = "<tr><th>Katman</th><th>Doğruluk</th><th>±</th><th>Makro-F1</th><th>Çoğunluk</th>"
    head += "<th>Kontrol</th><th>Seçicilik</th>" if has_ctrl else ""
    head += "<th>Eğitim/Doğr.</th></tr>"
    body = []
    for r in rows:
        cells = (f"<td>{_esc(r['layer'])}</td><td>{r['accuracy']:.3f}</td><td>{r['accuracy_std']:.3f}</td>"
                 f"<td>{r['macro_f1']:.3f}</td><td>{r['majority_baseline']:.3f}</td>")
        if has_ctrl:
            cells += f"<td>{r['control_accuracy']:.3f}</td><td>{r['selectivity']:+.3f}</td>"
        cells += f"<td>{r['num_train']}/{r['num_val']}</td>"
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table>{head}{"".join(body)}</table></div>'


def _moe_section(moe: Optional[dict]) -> str:
    if not moe:
        return ""
    parts = ['<h2 id="moe">MoE yönlendirme: uzman × morfolojik sınıf</h2>',
             '<p class="desc">Hücre: o sınıftaki token yönlendirmelerinin bu uzmana giden payı '
             '(sütun toplamı 1; top-k seçimlerin her biri sayılır). Koyu hücre = uzman o sınıfta yoğunlaşıyor. '
             'Eşit dağılım ≈ 1/uzman sayısı.</p><div class="grid">']
    classes = moe["classes"]
    for layer, d in moe["layers"].items():
        rows = []
        for e, (cnts, shares) in enumerate(zip(d["counts"], d["share_of_class"])):
            tds = "".join(
                f'<td style="background:color-mix(in srgb,var(--s1) {min(100, round(s * 100 * 1.6))}%,transparent)" '
                f'title="{_esc(f"uzman {e}, {classes[c]}: {cnts[c]} yönlendirme, pay {s:.1%}")}">{s:.0%}</td>'
                for c, s in enumerate(shares)
            )
            rows.append(f"<tr><td>uzman {e}</td>{tds}<td>{sum(cnts)}</td></tr>")
        head = "<tr><th>Uzman</th>" + "".join(f"<th>{_esc(c)}</th>" for c in classes) + "<th>Toplam</th></tr>"
        parts.append(f'<div class="card"><h3>{_esc(layer)} <span class="badge">top-{d["top_k"]}</span></h3>'
                     f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div></div>')
    parts.append("</div>")
    return "".join(parts)


_TOKEN_VIEW_JS = r"""
(function(){
const R=JSON.parse(document.getElementById('report-data').textContent);
const tv=R.token_view, feats=Object.keys(tv.probs);
const $=id=>document.getElementById(id);
if(!feats.length||!tv.sentences.length){$('tv').innerHTML='<p class="desc">Token görünümü için veri yok.</p>';return;}
const selS=$('tv-sent'),selF=$('tv-feat'),selL=$('tv-layer'),selC=$('tv-class');
tv.sentences.forEach((s,i)=>{const o=document.createElement('option');o.value=i;
 o.textContent=(i+1)+'. '+(s.text.length>70?s.text.slice(0,70)+'…':s.text)+' ['+s.split+']';selS.appendChild(o);});
feats.forEach(f=>{const o=document.createElement('option');o.value=f;o.textContent=f;selF.appendChild(o);});
function fillLayers(){const f=selF.value;const prev=selL.value;selL.innerHTML='';
 const best=R.features[f].best_layer;
 Object.keys(tv.probs[f]).forEach(l=>{const o=document.createElement('option');o.value=l;
  o.textContent=l+(l===best?' (en seçici)':'');selL.appendChild(o);});
 if([...selL.options].some(o=>o.value===prev))selL.value=prev;else if(best)selL.value=best;
 selC.innerHTML='';R.features[f].classes.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c;selC.appendChild(o);});
 selC.value=R.features[f].classes.length===2?1:0;
 $('tv-desc').textContent=R.features[f].description;}
const tip=$('tip');
function render(){const s=+selS.value,f=selF.value,l=selL.value,c=+selC.value;
 const sent=tv.sentences[s],P=tv.probs[f][l][s],gold=sent.labels[f],cls=R.features[f].classes;
 const box=$('tv-tokens');box.innerHTML='';
 sent.tokens.forEach((t,i)=>{const sp=document.createElement('span');sp.className='tok';
  const p=P[i];const pred=p.indexOf(Math.max(...p));
  sp.textContent=t.replace(/▁/g,'␣');
  sp.style.background='color-mix(in srgb,var(--s1) '+p[c]+'%,transparent)';
  if(gold[i]<0)sp.classList.add('nolabel');else if(pred!==gold[i])sp.classList.add('wrong');
  sp.addEventListener('mousemove',e=>{tip.style.display='block';tip.style.left=Math.min(e.clientX+12,innerWidth-290)+'px';tip.style.top=(e.clientY+14)+'px';
   tip.innerHTML='<b>'+t.replace(/[<>&]/g,'')+'</b><br>'+cls.map((n,k)=>n+': '+p[k]+'%').join('<br>')+'<br>gerçek: '+(gold[i]<0?'tanımsız':cls[gold[i]]);});
  sp.addEventListener('mouseleave',()=>tip.style.display='none');
  box.appendChild(sp);});
 $('tv-split').textContent=sent.split;}
selF.addEventListener('change',()=>{fillLayers();render();});
[selS,selL,selC].forEach(e=>e.addEventListener('change',render));
fillLayers();render();
})();
"""


def render_probe_html(report: dict) -> str:
    """Sonda raporunu bağımsız HTML'e dönüştür."""
    meta = report.get("metadata", {})
    body = [
        "<h1>Morfoloji Mikroskobu — Sonda Raporu</h1>",
        '<p class="lead">Her katmanın artık akışından Türkçe morfolojik özellikler doğrusal bir sondayla ne kadar '
        'okunabiliyor? Mavi çubuk: sonda doğruluğu (doğrulama cümleleri). Turuncu: kontrol görevi doğruluğu '
        '(token türüne rastgele etiket). Kesikli çizgi: çoğunluk sınıfı tabanı. Güvenilir bulgu = yüksek '
        '<b>seçicilik</b> (doğruluk − kontrol).</p>',
        '<div class="card">' + _meta_block(meta, ["checkpoint", "global_step", "num_sentences", "num_tokens", "site", "seeds", "epochs"]) + "</div>",
        '<nav class="meta"><a href="#katmanlar">Katman eğrileri</a><a href="#tv">Token görünümü</a>'
        + ('<a href="#moe">MoE yönlendirme</a>' if report.get("moe") else "") + "</nav>",
        '<h2 id="katmanlar">Katman başına sonda doğruluğu</h2>',
        '<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>sonda doğruluğu</span>'
        '<span><i class="sw" style="background:var(--s2)"></i>kontrol görevi</span>'
        '<span><i class="sw" style="background:none;border-top:2px dashed var(--text2);height:0"></i>çoğunluk tabanı</span></div>',
        '<div class="grid">',
    ]
    for name, f in report.get("features", {}).items():
        counts = ", ".join(f"{c}: {n}" for c, n in zip(f["classes"], f["class_counts"]))
        body.append(f'<section class="card feature" data-feature="{_esc(name)}"><h3>{_esc(name)}</h3>'
                    f'<p class="desc">{_esc(f["description"])}<br>Etiketli token: {f["num_labeled"]} ({_esc(counts)})')
        if f.get("best_layer"):
            body.append(f' · en seçici katman: <b>{_esc(f["best_layer"])}</b>')
        body.append("</p>")
        if f.get("rows"):
            body.append(layer_bar_svg(f["rows"]))
            body.append(f"<details><summary>Tablo</summary>{_probe_table(f['rows'])}</details>")
        else:
            body.append(f'<p class="desc">Atlandı: {_esc(f.get("skipped") or "veri yok")}</p>')
        body.append("</section>")
    body.append("</div>")
    body.append(
        '<h2 id="tv">Token görünümü</h2><div class="card">'
        '<p class="desc">Bir cümle, özellik, katman ve sınıf seçin: her token, sondanın o sınıfa verdiği olasılıkla '
        'renklenir. Kırmızı çerçeve = sondanın tahmini gerçek etiketten farklı; kesikli çerçeve = etiket tanımsız. '
        '"eğitim" cümlelerinde olasılıklar örneklem-içidir; dürüst değerlendirme için "doğrulama" cümlelerine bakın.</p>'
        '<div class="controls"><label>Cümle<select id="tv-sent"></select></label>'
        '<label>Özellik<select id="tv-feat"></select></label><label>Katman<select id="tv-layer"></select></label>'
        '<label>Renk sınıfı<select id="tv-class"></select></label><span class="badge" id="tv-split"></span></div>'
        '<p class="desc" id="tv-desc"></p><div class="tokens" id="tv-tokens"></div></div>'
    )
    body.append(_moe_section(report.get("moe")))
    body.append(_json_script(report, "report-data"))
    return _page("Sonda Raporu", "".join(body), _TOKEN_VIEW_JS)


def _ctx_html(c: dict) -> str:
    tok = c["token"].replace("▁", " ")
    return (f'<div class="ctx">{_esc(c["left"])}<mark>{_esc(tok)}</mark>{_esc(c["right"])} '
            f'<span class="desc">({c["activation"]:.2f})</span></div>')


def render_sae_html(report: dict) -> str:
    """SAE raporunu bağımsız HTML'e dönüştür."""
    meta, tr = report.get("metadata", {}), report.get("training", {})
    body = [
        "<h1>Morfoloji Mikroskobu — Seyrek Otokodlayıcı</h1>",
        '<p class="lead">TopK SAE, seçilen katmanın artık akışını seyrek latentlere ayırır. Her latent için en güçlü '
        'bağlamlar ve dilbilimsel etiketlerle en ilişkili latentler (nokta-çift serili korelasyon r) listelenir.</p>',
        '<div class="card">' + _meta_block({**meta, **tr}, ["checkpoint", "layer", "num_tokens", "d_in", "d_hidden", "k"])
        + _meta_block({"başlangıç NMSE": f'{tr.get("initial_nmse", float("nan")):.3f}', "son NMSE": f'{tr.get("nmse", float("nan")):.3f}',
                       "açıklanan varyans": f'{tr.get("fve", float("nan")):.3f}', "ölü latent oranı": f'{tr.get("dead_fraction", float("nan")):.3f}'},
                      ["başlangıç NMSE", "son NMSE", "açıklanan varyans", "ölü latent oranı"]) + "</div>",
        '<h2 id="iliski">Etiketlerle en ilişkili latentler</h2><div class="grid">',
    ]
    contexts = {str(f["feature"]): f["contexts"] for f in report.get("top_features", [])}
    contexts.update(report.get("feature_contexts", {}))
    for name, by_class in report.get("associations", {}).items():
        if not by_class:
            continue
        body.append(f'<section class="card"><h3>{_esc(name)}</h3>')
        for cname, ranked in by_class.items():
            rows = "".join(
                f'<tr><td>#{r["feature"]}</td><td>{r["r"]:+.3f}</td><td>{r["fire_rate_pos"]:.0%}</td><td>{r["fire_rate_neg"]:.0%}</td></tr>'
                for r in ranked
            )
            body.append(f'<p class="desc">sınıf <b>{_esc(cname)}</b></p><div class="table-wrap"><table>'
                        f'<tr><th>Latent</th><th>r</th><th>ateş (sınıf)</th><th>ateş (diğer)</th></tr>{rows}</table></div>')
            best = ranked[0]
            if str(best["feature"]) in contexts and contexts[str(best["feature"])]:
                body.append(f'<details><summary>#{best["feature"]} bağlamları</summary>'
                            + "".join(_ctx_html(c) for c in contexts[str(best["feature"])]) + "</details>")
        body.append("</section>")
    body.append('</div><h2 id="latentler">En sık ateşlenen latentler</h2><div class="grid">')
    for f in report.get("top_features", []):
        body.append(f'<section class="card"><h3>Latent #{f["feature"]} <span class="badge">{f["fires"]} token</span></h3>'
                    + "".join(_ctx_html(c) for c in f["contexts"]) + "</section>")
    body.append("</div>")
    body.append(_json_script(report, "report-data"))
    return _page("SAE Raporu", "".join(body))


def render_compare_html(report: dict) -> str:
    """Karşılaştırma raporunu bağımsız HTML'e dönüştür."""
    a, b = report["names"]
    body = [
        "<h1>Morfoloji Mikroskobu — Checkpoint Karşılaştırması</h1>",
        f'<p class="lead">Katman başına sonda doğruluğu farkı: <b>{_esc(b)}</b> − <b>{_esc(a)}</b>. '
        'Mavi = B daha iyi okunuyor, kırmızı = daha kötü. Kalın çerçeve ve * : |Δ| &gt; 2·SE (kaba gürültü eşiği; '
        'binom SE + tohum sapması). Aynı metinler, aynı tohumlar kullanılmalıdır.</p>',
    ]
    for w in report.get("warnings", []):
        body.append(f'<div class="warn">{_esc(w)}</div>')
    body.append('<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>artış</span>'
                '<span><i class="sw" style="background:var(--neg)"></i>azalış</span></div><div class="grid">')
    for name, f in report.get("features", {}).items():
        rows = "".join(
            f'<tr><td>{_esc(r["layer"])}</td><td>{r["accuracy_a"]:.3f}</td><td>{r["accuracy_b"]:.3f}</td>'
            f'<td class="{"pos" if r["delta_accuracy"] >= 0 else "negv"}">{r["delta_accuracy"]:+.3f}{" *" if r["notable"] else ""}</td>'
            f'<td>{r["se"]:.3f}</td><td>{r.get("delta_selectivity", float("nan")):+.3f}</td></tr>'
            for r in f["rows"]
        )
        body.append(
            f'<section class="card feature" data-feature="{_esc(name)}"><h3>{_esc(name)}</h3>'
            f'<p class="desc">{_esc(f["description"])}<br>Ortalama Δ doğruluk: <b>{f["mean_delta_accuracy"]:+.3f}</b></p>'
            + delta_bar_svg(f["rows"])
            + f'<details><summary>Tablo</summary><div class="table-wrap"><table><tr><th>Katman</th><th>{_esc(a)}</th>'
              f'<th>{_esc(b)}</th><th>Δ</th><th>SE</th><th>Δ seçicilik</th></tr>{rows}</table></div></details></section>'
        )
    body.append("</div>")
    body.append(_json_script(report, "report-data"))
    return _page("Karşılaştırma Raporu", "".join(body))
