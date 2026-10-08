# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — lm-evaluation-harness adaptörü

ToprakLM checkpointlerini EleutherAI lm-evaluation-harness ile standart
benchmarklarda (TurkishMMLU, Belebele, XNLI, XCOPA, ...) değerlendirmek için
`lm_eval.api.model.LM` arayüzünü uygular.

lm_eval CLI üzerinden de kullanılabilir (`--model toprak`), ancak tercih
edilen giriş noktası `evaluation/run_lm_eval.py` scriptidir.
"""

import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


@register_model("toprak")
class ToprakLMEval(LM):
    """
    lm-eval için ToprakLM sarmalayıcısı.

    Args:
        checkpoint: .pt checkpoint yolu (model verilmezse zorunlu)
        tokenizer: SentencePiece .model yolu veya ToprakTokenizer nesnesi
        model: Önceden yüklenmiş ToprakLM (testler / eğitim içi değerlendirme)
        device: "cuda" / "mps" / "cpu"; None ise otomatik algılanır
        batch_size: loglikelihood istekleri için batch boyutu
        max_length: Bağlam penceresi; None ise config.max_seq_len
        max_gen_toks: generate_until için varsayılan yeni token limiti
        dtype: CUDA'da autocast tipi ("bfloat16", "float16") veya "float32"
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        tokenizer="toprak_tokenizer.model",
        model=None,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: Optional[int] = None,
        max_gen_toks: int = 256,
        dtype: str = "float32",
    ):
        super().__init__()

        if device is None:
            from model.config import detect_device
            device = detect_device()
        self._device = device

        if isinstance(tokenizer, str):
            from model.tokenizer import ToprakTokenizer
            tokenizer = ToprakTokenizer(tokenizer)
        self.tokenizer = tokenizer

        if model is None:
            if checkpoint is None:
                raise ValueError("checkpoint veya model verilmelidir")
            from evaluation.evaluate_suite import load_checkpoint_model
            model, _, _ = load_checkpoint_model(checkpoint, tokenizer, device)
        self.model = model.to(device).eval()

        self.batch_size = int(batch_size)
        self.max_length = int(max_length or self.model.config.max_seq_len)
        self.max_gen_toks = int(max_gen_toks)

        amp_dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        if dtype not in ("float32", *amp_dtypes):
            raise ValueError(f"Desteklenmeyen dtype: {dtype}")
        # MPS autocast NaN ürettiği için yalnız CUDA'da karışık hassasiyet.
        self.amp_dtype = amp_dtypes.get(dtype) if str(device).startswith("cuda") else None

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        from lm_eval.utils import simple_parse_args_string
        args = simple_parse_args_string(arg_string)
        extra = {k: v for k, v in (additional_config or {}).items() if v is not None}
        if "device" in extra:
            args.setdefault("device", extra["device"])
        if "batch_size" in extra and str(extra["batch_size"]).isdigit():
            args.setdefault("batch_size", int(extra["batch_size"]))
        return cls(**args)

    # ── Yardımcılar ──────────────────────────────────────

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return getattr(self.tokenizer, "pad_token_id", 0)

    def tok_encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_bos=False, add_eos=False)

    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        """
        SentencePiece sınırda farklı birleştirme yapabildiği için bağlam ve
        devam metni birlikte kodlanıp bağlam uzunluğundan bölünür
        (HF adaptörüyle aynı strateji). Bağlam sonundaki boşluklar devam
        metnine taşınır.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        continuation_enc = whole[len(context_enc):]
        if not continuation_enc:
            # Sınır birleşmesi devamı yuttuysa ayrı kodla.
            continuation_enc = self.tok_encode(continuation)
        return [self.bos_token_id] + context_enc, continuation_enc

    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                logits, _, _ = self.model(input_ids)
        else:
            logits, _, _ = self.model(input_ids)
        return logits.float()

    @torch.no_grad()
    def _score_batch(
        self, items: List[Tuple[List[int], List[int]]]
    ) -> List[Tuple[float, bool]]:
        """
        (bağlam_tokenları, devam_tokenları) çiftleri için devamın toplam
        log-olasılığını ve açgözlü (greedy) eşleşmeyi hesaplar.
        """
        inputs, cont_lens = [], []
        for context_enc, continuation_enc in items:
            full = (context_enc + continuation_enc)[-(self.max_length + 1):]
            inputs.append(full[:-1])
            cont_lens.append(len(continuation_enc))

        width = max(len(x) for x in inputs)
        # Sağdan padding: causal attention gerçek tokenları etkilemez.
        batch = torch.full((len(inputs), width), self.pad_token_id, dtype=torch.long)
        for i, ids in enumerate(inputs):
            batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        log_probs = F.log_softmax(self._forward_logits(batch.to(self.device)), dim=-1)

        results = []
        for i, ((context_enc, continuation_enc), n_cont) in enumerate(zip(items, cont_lens)):
            end = len(inputs[i])
            # Devam tokenı, bağlam penceresinden büyükse en fazla pencere kadar skorlanır.
            n_cont = min(n_cont, end)
            cont_targets = torch.tensor(continuation_enc[-n_cont:], device=self.device)
            cont_logp = log_probs[i, end - n_cont : end]
            greedy = bool((cont_logp.argmax(dim=-1) == cont_targets).all())
            total = float(cont_logp.gather(-1, cont_targets.unsqueeze(-1)).sum())
            results.append((total, greedy))
        return results

    def _loglikelihood_tokens(
        self, items: List[Tuple[List[int], List[int]]], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        # Uzundan kısaya sıralama: padding israfını azaltır, OOM'u başta gösterir.
        order = sorted(
            range(len(items)),
            key=lambda i: -len(items[i][0]) - len(items[i][1]),
        )
        results: List[Optional[Tuple[float, bool]]] = [None] * len(items)
        for start in tqdm(
            range(0, len(order), self.batch_size),
            disable=disable_tqdm or self.rank != 0,
            desc="loglikelihood",
        ):
            chunk = order[start : start + self.batch_size]
            for idx, res in zip(chunk, self._score_batch([items[i] for i in chunk])):
                results[idx] = res
        return results

    # ── lm-eval arayüzü ──────────────────────────────────

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        items = [self._encode_pair(*req.args) for req in requests]
        results = self._loglikelihood_tokens(items, disable_tqdm=disable_tqdm)
        for req, res in zip(requests, results):
            self.cache_hook.add_partial("loglikelihood", req.args, res)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        outputs = []
        for req in tqdm(requests, disable=disable_tqdm or self.rank != 0, desc="rolling"):
            (text,) = req.args
            windows = [
                make_disjoint_window(window)
                for window in get_rolling_token_windows(
                    token_list=self.tok_encode(text),
                    prefix_token=self.bos_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )
            ]
            scores = self._loglikelihood_tokens(windows, disable_tqdm=True)
            total = sum(score for score, _ in scores)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), total)
            outputs.append(total)
        return outputs

    @torch.no_grad()
    def _generate(self, context: str, until: List[str], max_gen_toks: int,
                  temperature: float, do_sample: bool) -> str:
        context_enc = [self.bos_token_id] + self.tok_encode(context)
        budget = max(self.max_length - max_gen_toks, 1)
        context_enc = context_enc[-budget:]
        max_gen_toks = min(max_gen_toks, self.max_length - len(context_enc))

        input_ids = torch.tensor([context_enc], dtype=torch.long, device=self.device)
        generated: List[int] = []
        past_kvs = None
        text = ""
        for _ in range(max_gen_toks):
            step_input = input_ids if past_kvs is None else input_ids[:, -1:]
            logits, _, past_kvs = self.model(step_input, past_kvs=past_kvs)
            logits = logits[:, -1, :].float()
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            token_id = int(next_token.item())
            if token_id == self.eos_token_id:
                break
            generated.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            text = self.tokenizer.decode(generated)
            if any(stop in text for stop in until):
                break

        text = self.tokenizer.decode(generated)
        for stop in until:
            if stop:
                text = text.split(stop)[0]
        return text

    def generate_until(self, requests, disable_tqdm: bool = False):
        outputs = []
        for req in tqdm(requests, disable=disable_tqdm or self.rank != 0, desc="generate"):
            context, gen_kwargs = req.args
            gen_kwargs = dict(gen_kwargs or {})
            until = gen_kwargs.pop("until", None) or []
            if isinstance(until, str):
                until = [until]
            max_gen_toks = int(gen_kwargs.pop("max_gen_toks", self.max_gen_toks))
            temperature = float(gen_kwargs.pop("temperature", 0.0))
            do_sample = bool(gen_kwargs.pop("do_sample", False))

            text = self._generate(context, until, max_gen_toks, temperature, do_sample)
            self.cache_hook.add_partial("generate_until", (context, req.args[1]), text)
            outputs.append(text)
        return outputs
