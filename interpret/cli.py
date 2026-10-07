# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Morfoloji Mikroskobu komut satırı

Alt komutlar:
    probe    aktivasyon topla → tüm özellikler × tüm katmanlar sonda →
             probe.json + probe.html
    sae      bir katmanda TopK SAE eğit → sae.json + sae.html
    compare  iki checkpoint (veya iki probe.json) için katman bazında
             sonda doğruluğu farkı → compare.json + compare.html

Örnekler:
    python -m interpret.cli probe --checkpoint checkpoints/toprak_last.pt \\
        --tokenizer toprak_tokenizer.model --texts interpret/examples/sentences.txt \\
        --out reports/probe
    python -m interpret.cli sae --checkpoint ckpt.pt --layer L6 --out reports/sae
    python -m interpret.cli compare --checkpoint-a runs/baseline.pt \\
        --checkpoint-b runs/vowel.pt --name-a baseline --name-b vowel_harmony --out reports/cmp
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from typing import List, Optional

import torch

from interpret.activations import collect
from interpret.features import FEATURES
from interpret.report import (
    build_compare_report,
    build_probe_report,
    build_sae_report,
    render_compare_html,
    render_probe_html,
    render_sae_html,
    write_report,
)

DEFAULT_TEXTS = os.path.join(os.path.dirname(__file__), "examples", "sentences.txt")


def read_texts(path: str) -> List[str]:
    """Satır başına bir cümle; boş satırlar ve '#' ile başlayanlar atlanır."""
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def _sha256(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_model(checkpoint_path: str, tokenizer, device: str = "cpu"):
    """
    Checkpoint'ten ToprakLM kur. Config'te ModelConfig'te olmayan alanlar
    atlanır; tokenizer token sınıfı tablosu için modele verilir.

    Returns:
        (model, config, global_step)
    """
    from model.config import ModelConfig
    from model.transformer import ToprakLM

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = {k: v for k, v in (ckpt.get("config") or {}).items() if k in fields}
    cfg["device"] = device
    config = ModelConfig(**cfg)
    model = ToprakLM(config, tokenizer=tokenizer).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "freqs_cis"]
    if missing:
        print(f"Uyarı: checkpoint'te eksik ağırlıklar: {missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)
    if unexpected:
        print(f"Uyarı: beklenmeyen ağırlıklar atlandı: {unexpected[:5]}", file=sys.stderr)
    model.eval()
    return model, config, int(ckpt.get("global_step", 0) or 0)


def _metadata(args, checkpoint: str, config, step: int) -> dict:
    return {
        "checkpoint": checkpoint,
        "global_step": step,
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": _sha256(args.tokenizer),
        "texts_file": args.texts,
        "model_config": {k: v for k, v in config.architecture_dict().items() if k != "rope_scaling"},
    }


def _probe_one(args, checkpoint: str, tokenizer, texts: List[str]) -> dict:
    model, config, step = load_checkpoint_model(checkpoint, tokenizer, args.device)
    collected = collect(model, tokenizer, texts, max_len=args.max_len, sites=(args.site,), device=args.device)
    return build_probe_report(
        collected,
        features=args.features,
        site=args.site,
        seeds=args.seeds,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        view_sentences=args.view_sentences,
        metadata=_metadata(args, checkpoint, config, step),
        num_experts=config.num_experts or None,
    )


def _tokenizer(path: str):
    from model.tokenizer import ToprakTokenizer
    return ToprakTokenizer(path)


def cmd_probe(args) -> dict:
    tokenizer = _tokenizer(args.tokenizer)
    report = _probe_one(args, args.checkpoint, tokenizer, read_texts(args.texts))
    paths = write_report(args.out, "probe", report, render_probe_html(report))
    _print_probe_summary(report)
    print(f"Rapor: {paths['html']}")
    return paths


def _print_probe_summary(report: dict):
    for name, f in report["features"].items():
        if not f.get("rows"):
            print(f"  {name:<18} atlandı ({f.get('skipped')})")
            continue
        best = max(f["rows"], key=lambda r: r.get("selectivity", r["accuracy"]))
        sel = f" seçicilik={best['selectivity']:+.3f}" if "selectivity" in best else ""
        print(f"  {name:<18} en iyi {best['layer']:<4} doğruluk={best['accuracy']:.3f} "
              f"çoğunluk={best['majority_baseline']:.3f}{sel}")


def _resolve_layer(spec: str, num_layers: int):
    s = spec.strip()
    if s.lower() in ("emb", "embed", "0"):
        return "emb", None
    if s.upper().startswith("L"):
        s = s[1:]
    i = int(s)
    if not 1 <= i <= num_layers:
        raise SystemExit(f"--layer 1..{num_layers} aralığında olmalı (veya 'emb')")
    return f"L{i}", i - 1


def cmd_sae(args) -> dict:
    from interpret.sae import train_sae

    tokenizer = _tokenizer(args.tokenizer)
    model, config, step = load_checkpoint_model(args.checkpoint, tokenizer, args.device)
    layer_name, block = _resolve_layer(args.layer, config.num_layers)
    collected = collect(
        model, tokenizer, read_texts(args.texts), max_len=args.max_len, sites=(args.site,),
        layers=[block] if block is not None else [0], device=args.device,
    )
    acts = collected.acts["embed"][0] if block is None else collected.acts[args.site][block]
    d_hidden = args.d_hidden or 8 * config.d_model
    print(f"SAE eğitimi: katman={layer_name}, token={len(acts)}, d_hidden={d_hidden}, k={args.k}")
    result = train_sae(acts, d_hidden=d_hidden, k=args.k, epochs=args.epochs, lr=args.lr, seed=args.seed,
                       batch_size=args.batch_size)
    print(f"  NMSE {result.initial_nmse:.3f} → {result.nmse:.3f}, FVE={result.fve:.3f}, ölü={result.dead_fraction:.2%}")
    meta = _metadata(args, args.checkpoint, config, step)
    meta["site"] = args.site
    report = build_sae_report(result, collected, acts, layer_name, top_features=args.top_features, metadata=meta)
    paths = write_report(args.out, "sae", report, render_sae_html(report))
    if args.save_sae:
        torch.save({"state_dict": result.sae.state_dict(), "d_in": result.sae.d_in, "d_hidden": d_hidden,
                    "k": args.k, "layer": layer_name, "site": args.site},
                   os.path.join(args.out, "sae.pt"))
    print(f"Rapor: {paths['html']}")
    return paths


def cmd_compare(args) -> dict:
    if args.report_a and args.report_b:
        with open(args.report_a, encoding="utf-8") as fh:
            rep_a = json.load(fh)
        with open(args.report_b, encoding="utf-8") as fh:
            rep_b = json.load(fh)
    elif args.checkpoint_a and args.checkpoint_b:
        tokenizer = _tokenizer(args.tokenizer)
        texts = read_texts(args.texts)
        rep_a = _probe_one(args, args.checkpoint_a, tokenizer, texts)
        rep_b = _probe_one(args, args.checkpoint_b, tokenizer, texts)
        write_report(args.out, f"probe_{args.name_a}", rep_a, render_probe_html(rep_a))
        write_report(args.out, f"probe_{args.name_b}", rep_b, render_probe_html(rep_b))
    else:
        raise SystemExit("compare: --checkpoint-a/--checkpoint-b ya da --report-a/--report-b verin")
    report = build_compare_report(rep_a, rep_b, args.name_a, args.name_b)
    paths = write_report(args.out, "compare", report, render_compare_html(report))
    for w in report["warnings"]:
        print(f"Uyarı: {w}")
    for name, f in report["features"].items():
        notable = [r["layer"] for r in f["rows"] if r["notable"]]
        print(f"  {name:<18} ortalama Δ={f['mean_delta_accuracy']:+.3f}  belirgin katmanlar: {', '.join(notable) or '-'}")
    print(f"Rapor: {paths['html']}")
    return paths


def _add_common(p, checkpoint: bool = True):
    if checkpoint:
        p.add_argument("--checkpoint", required=True, help="Model checkpoint dosyası (.pt)")
    p.add_argument("--tokenizer", default="toprak_tokenizer.model", help="SentencePiece tokenizer modeli")
    p.add_argument("--texts", default=DEFAULT_TEXTS, help="Satır başına bir cümle içeren metin dosyası")
    p.add_argument("--out", required=True, help="Rapor dizini")
    p.add_argument("--max-len", type=int, default=128, help="Cümle başına en çok token")
    p.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    p.add_argument("--site", default="resid_post", choices=["resid_post", "attn_out", "ffn_out"],
                   help="İncelenecek aktivasyon noktası")


def _add_probe_opts(p):
    p.add_argument("--features", nargs="+", choices=list(FEATURES), default=None,
                   help="Sondalanacak özellikler (varsayılan: hepsi)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="Sonda tohumları (birden çoksa ort. ± sapma)")
    p.add_argument("--epochs", type=int, default=200, help="Sonda eğitim adımı (tam-batch)")
    p.add_argument("--lr", type=float, default=0.05, help="Sonda öğrenme oranı")
    p.add_argument("--weight-decay", type=float, default=1e-3, help="Sonda L2 katsayısı")
    p.add_argument("--val-split", type=float, default=0.25, help="Doğrulama cümle oranı")
    p.add_argument("--view-sentences", type=int, default=40, help="Token görünümüne gömülecek cümle sayısı")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m interpret.cli", description="Toprak — Morfoloji Mikroskobu")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="Tüm katmanlarda morfolojik özellik sondası")
    _add_common(p)
    _add_probe_opts(p)
    p.set_defaults(func=cmd_probe)

    s = sub.add_parser("sae", help="Bir katmanda TopK seyrek otokodlayıcı eğit")
    _add_common(s)
    s.add_argument("--layer", required=True, help="Katman: 'emb' veya 1 tabanlı blok (ör. L6 ya da 6)")
    s.add_argument("--d-hidden", type=int, default=None, help="Latent sayısı (varsayılan 8 × d_model)")
    s.add_argument("--k", type=int, default=16, help="Token başına aktif latent sayısı")
    s.add_argument("--epochs", type=int, default=30, help="SAE epok sayısı")
    s.add_argument("--lr", type=float, default=1e-3, help="SAE öğrenme oranı")
    s.add_argument("--batch-size", type=int, default=256, help="SAE batch boyutu")
    s.add_argument("--seed", type=int, default=0, help="Tohum")
    s.add_argument("--top-features", type=int, default=24, help="Raporlanacak latent sayısı")
    s.add_argument("--save-sae", action="store_true", help="SAE ağırlıklarını out/sae.pt olarak kaydet")
    s.set_defaults(func=cmd_sae)

    c = sub.add_parser("compare", help="İki checkpoint'in sonda doğruluklarını karşılaştır")
    _add_common(c, checkpoint=False)
    c.add_argument("--checkpoint-a", help="Referans checkpoint (ör. ablation baseline)")
    c.add_argument("--checkpoint-b", help="Aday checkpoint (ör. vowel_harmony)")
    c.add_argument("--report-a", help="Hazır probe.json (checkpoint yerine)")
    c.add_argument("--report-b", help="Hazır probe.json (checkpoint yerine)")
    c.add_argument("--name-a", default="baseline", help="A'nın görünen adı")
    c.add_argument("--name-b", default="aday", help="B'nin görünen adı")
    _add_probe_opts(c)
    c.set_defaults(func=cmd_compare)
    return parser


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
