# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Anayasaya Dayalı Öz-Düzeltme (Constitutional AI)

Bir prompt için:
  1. İlk cevap üretilir,
  2. Toprak Anayasası'ndan rastgele N ilke seçilir,
  3. Her ilke için model önce cevabı eleştirir, sonra eleştiriye göre düzeltir.

Son düzeltme "chosen", ilk cevap "rejected" olarak `training/dpo.py` ile
uyumlu tercih çiftine dönüşür; düzeltilmiş cevaplar ayrıca SFT kaydı olarak
yazılabilir.

`generate_fn(messages) -> str` herhangi bir model çağrısı olabilir: Toprak'ın
kendisi (öz-düzeltme) ya da lisansı uygun harici bir öğretmen model.

Kullanım:
    python alignment/constitutional.py --checkpoint checkpoints/toprak_sft.pt \\
        --prompts alignment/examples/prompts.txt --output data/cai_pairs.jsonl \\
        --sft-output data/cai_sft.jsonl --num-principles 2
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

CONSTITUTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constitution.json")

Messages = List[Dict[str, str]]
GenerateFn = Callable[[Messages], str]


@dataclass(frozen=True)
class Principle:
    """Anayasanın tek bir ilkesi."""

    id: str
    title: str
    critique_prompt: str
    revision_prompt: str
    description: str = ""


def load_principles(path: str = CONSTITUTION_PATH) -> List[Principle]:
    """constitution.json dosyasından ilkeleri yükle."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["principles"] if isinstance(data, dict) else data
    return [
        Principle(
            id=item["id"],
            title=item["title"],
            critique_prompt=item["critique_prompt"],
            revision_prompt=item["revision_prompt"],
            description=item.get("description", ""),
        )
        for item in items
    ]


def _prompt_messages(prompt: Union[str, Sequence[Dict[str, str]]]) -> Messages:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(m) for m in prompt]


class ConstitutionalReviser:
    """
    Eleştir → düzelt döngüsünü yürütür.

    Args:
        generate_fn: messages -> cevap metni
        principles: İlke listesi (varsayılan: constitution.json)
        rng: random.Random örneği (tekrarlanabilirlik için)
        num_principles: Prompt başına uygulanacak ilke sayısı
        critique_prefix / revision_prefix: Eleştiri ve düzeltme isteklerinin öneki
    """

    def __init__(
        self,
        generate_fn: GenerateFn,
        principles: Optional[Sequence[Principle]] = None,
        rng: Optional[random.Random] = None,
        num_principles: int = 2,
        critique_prefix: str = "Eleştiri isteği: ",
        revision_prefix: str = "Düzeltme isteği: ",
    ):
        self.generate_fn = generate_fn
        self.principles = list(principles) if principles is not None else load_principles()
        if not self.principles:
            raise ValueError("En az bir ilke gerekli")
        self.rng = rng or random.Random(0)
        self.num_principles = num_principles
        self.critique_prefix = critique_prefix
        self.revision_prefix = revision_prefix

    def sample_principles(self, n: Optional[int] = None) -> List[Principle]:
        n = min(n or self.num_principles, len(self.principles))
        return self.rng.sample(self.principles, n)

    def revise(
        self,
        prompt: Union[str, Sequence[Dict[str, str]]],
        principles: Optional[Sequence[Principle]] = None,
        initial: Optional[str] = None,
    ) -> dict:
        """
        Tek bir prompt için öz-düzeltme yörüngesi üret.

        Returns:
            {"prompt", "initial", "final", "steps": [{"principle", "critique", "revision"}]}
        """
        base = _prompt_messages(prompt)
        current = initial if initial is not None else self.generate_fn(base).strip()
        first = current
        steps = []
        for principle in principles if principles is not None else self.sample_principles():
            convo = base + [
                {"role": "assistant", "content": current},
                {"role": "user", "content": self.critique_prefix + principle.critique_prompt},
            ]
            critique = self.generate_fn(convo).strip()
            convo = convo + [
                {"role": "assistant", "content": critique},
                {"role": "user", "content": self.revision_prefix + principle.revision_prompt},
            ]
            revision = self.generate_fn(convo).strip()
            if revision:
                current = revision
            steps.append({"principle": principle.id, "critique": critique, "revision": revision})
        return {"prompt": prompt, "initial": first, "final": current, "steps": steps}


def build_preference_pairs(
    prompts: Iterable[Union[str, Sequence[Dict[str, str]]]],
    reviser: ConstitutionalReviser,
    keep_unchanged: bool = False,
    trajectories: Optional[list] = None,
) -> List[dict]:
    """
    Her prompt için {"prompt", "chosen", "rejected", "principles"} üret.
    chosen = son düzeltme, rejected = ilk cevap. Düzeltme ilk cevapla aynıysa
    (öğrenilecek fark yok) çift atlanır (keep_unchanged=False).

    trajectories listesi verilirse tüm yörüngeler ona eklenir.
    """
    pairs = []
    for prompt in prompts:
        traj = reviser.revise(prompt)
        if trajectories is not None:
            trajectories.append(traj)
        if not traj["final"] or (traj["final"] == traj["initial"] and not keep_unchanged):
            continue
        pairs.append({
            "prompt": traj["prompt"],
            "chosen": traj["final"],
            "rejected": traj["initial"],
            "principles": [s["principle"] for s in traj["steps"]],
        })
    return pairs


def build_sft_records(pairs: Iterable[dict]) -> List[dict]:
    """Tercih çiftlerinden düzeltilmiş cevaplarla SFT kayıtları üret."""
    records = []
    for pair in pairs:
        messages = _prompt_messages(pair["prompt"])
        records.append({"messages": messages + [{"role": "assistant", "content": pair["chosen"]}]})
    return records


def read_prompts(path: str) -> List[str]:
    """Satır başına bir prompt; boş satırlar ve '#' ile başlayanlar atlanır."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def make_toprak_generate_fn(
    model,
    tokenizer,
    device: str = "cpu",
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 4,
) -> GenerateFn:
    """Toprak modelini sohbet şablonuyla generate_fn'e sar."""
    from inference.chat import fit_history
    from inference.generate import generate_text
    from model.chat_template import ChatTemplate

    template = ChatTemplate(tokenizer, system_prompt=system_prompt)
    max_positions = model.freqs_cis.size(0)

    def generate_fn(messages: Messages) -> str:
        _, prompt_ids, new_tokens = fit_history(template, messages, max_positions, max_new_tokens)
        return generate_text(
            model, tokenizer, prompt="", max_new_tokens=new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size, device=device,
            stop_ids=template.stop_ids, prompt_ids=prompt_ids, return_new_only=True,
        )

    return generate_fn


def main(argv=None):
    import argparse

    from training.sft import write_jsonl

    parser = argparse.ArgumentParser(
        description="🌱 Toprak — Anayasaya dayalı öz-düzeltme ile tercih verisi üretimi"
    )
    parser.add_argument("--checkpoint", required=True, help="Cevap üretecek model checkpoint'i")
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model", help="Tokenizer model dosyası")
    parser.add_argument("--prompts", required=True, help="Satır başına bir prompt içeren metin dosyası")
    parser.add_argument("--output", required=True, help="DPO tercih çiftleri (JSONL)")
    parser.add_argument("--sft-output", default=None, help="Düzeltilmiş cevaplarla SFT kayıtları (JSONL)")
    parser.add_argument("--trajectories", default=None, help="Tüm eleştiri/düzeltme yörüngeleri (JSONL)")
    parser.add_argument("--constitution", default=CONSTITUTION_PATH, help="Anayasa JSON dosyası")
    parser.add_argument("--num-principles", type=int, default=2, help="Prompt başına ilke sayısı")
    parser.add_argument("--system", default=None, help="Sistem mesajı")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default=None, help="Cihaz (varsayılan: otomatik)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-unchanged", action="store_true",
                        help="Düzeltmenin değiştirmediği cevapları da yaz")
    args = parser.parse_args(argv)

    import torch

    from inference.generate import load_model
    from model.config import detect_device
    from model.tokenizer import ToprakTokenizer
    from utils.validation import setup_error_handler, validate_checkpoint, validate_tokenizer

    setup_error_handler()
    validate_checkpoint(args.checkpoint)
    validate_tokenizer(args.tokenizer)
    device = args.device or detect_device()
    torch.manual_seed(args.seed)

    model, _ = load_model(args.checkpoint, device)
    tokenizer = ToprakTokenizer(args.tokenizer)
    generate_fn = make_toprak_generate_fn(
        model, tokenizer, device=device, system_prompt=args.system,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    reviser = ConstitutionalReviser(
        generate_fn, load_principles(args.constitution),
        rng=random.Random(args.seed), num_principles=args.num_principles,
    )
    prompts = read_prompts(args.prompts)
    print(f"🌱 Toprak — öz-düzeltme: {len(prompts)} prompt, prompt başına {args.num_principles} ilke")
    trajectories: list = []
    pairs = build_preference_pairs(prompts, reviser, keep_unchanged=args.keep_unchanged,
                                   trajectories=trajectories)
    write_jsonl(args.output, pairs)
    print(f"  ✓ {len(pairs)} tercih çifti → {args.output}")
    if args.sft_output:
        write_jsonl(args.sft_output, build_sft_records(pairs))
        print(f"  ✓ SFT kayıtları → {args.sft_output}")
    if args.trajectories:
        write_jsonl(args.trajectories, trajectories)
        print(f"  ✓ Yörüngeler → {args.trajectories}")


if __name__ == "__main__":
    main()
