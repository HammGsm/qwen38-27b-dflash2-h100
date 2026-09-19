# Qwen3.8-27B + DFlash2 drafter on H100: +82% decode, and the vLLM fix quantized drafters need

**+82% decode on one H100 PCIe: Qwen3.8-27B + a 1.19 GB DFlash2 block drafter — and the 39-line patch vLLM 0.28/0.29 still needs.**

Apache-2.0 · community benchmark report · no affiliation with vLLM, Qwen or the model authors.

| | |
|---|---|
| Reproduce it | `MAX_TOKENS=1024 python3 bench.py http://127.0.0.1:8000/v1/chat/completions 5` |
| The fix | [`dflash2-quantized-drafter-kv.patch`](./dflash2-quantized-drafter-kv.patch) — sha256 `a13c444641e6a7e5cf39dc3355cf3a5f3c208beb4e4328ce5620d4791a236c47` |
| Headline numbers | 136.9 → **249.7 tok/s** (TP=1, +82%) · **316.3 tok/s** (TP=2, +131%) |

---

## 1. Hardware / components actually used

| Component | Value |
|---|---|
| GPU | **2x NVIDIA H100 PCIe 80 GB** (`memory.total 81559 MiB`, `compute_cap 9.0`) |
| Interconnect | **PCIe only, no NVLink** — `nvidia-smi topo -m` reports `SYS` between the GPUs (crosses the SMP interconnect) |
| PCIe link | Gen **5** x16 max (Gen 5 x16 current) |
| CPU | **2x AMD EPYC 9654 96-Core** (192 cores, 1 thread/core) |
| RAM | **755 GB** total (732 GB free) — only *reserve 64 GB* for vLLM; it actually peaks at **3.3 GB RSS** (measured: `VmHWM 3,325,436 kB`) |
| Driver / CUDA | **580.178.04 / CUDA 13.0** |
| Python | **3.11.16** (conda env) |
| vLLM | **0.28.0** |
| PyTorch | **2.13.0+cu130** |
| transformers | **5.16.1** |
| Scheduler | Slurm: the job *reserves* `gpu:1 + 32 CPUs + 64 GB`, but `nproc` inside the allocation reports **32** vs 192 on the host — reserve what the scheduler wants, measure what you actually use |

## 2. Models

**Target — Qwen3.8-27B, INT4 GPTQ (19 GB)**
- Lineage: `llmfan46/Qwen3.8-27B-Ultra-Uncensored-Heretic-Native-MTP-Preserved` → GPTQ repack, MTP heads preserved.
- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`, **vision tower present and working**, MTP preserved.
- Quantization: `method: gptq`, `bits: 4`, `group_size: 128`, `desc_act: false`, `checkpoint_format: gptq`, `lm_head: false`, built with `gptqmodel:7.3.5`. Kernel vLLM selects: **`MacheteLinearKernel`** (W4A16).
- The vision tower is why we keep `--language-model-only` **off**: dropping it saves VRAM but **breaks image input** (tested).

**Drafter — DFlash2, W4A16 compressed-tensors (1.19 GB)**
- `incoai/Qwen3.8-27B-DFlash2` is the bf16 original (3.85 GB, 5 Qwen3-style layers, 1.92 B params); `syv-ai/HyperQwen` repacked it to **W4A16 compressed-tensors** (pack-quantized, group 128, symmetric, Marlin, GPTQ) → **1.19 GB**, which is what lets it sit next to a 27B target on a 24 GB card.
- `dflash_config`: block_size 8, conv_group_size 16, conv_kernel_size 2, selector_rank 256, selector_top_k 16, target_layer_ids [5, 19, 33, 47, 61].

## 3. Results (single-stream greedy decode, median of n=5)

| Config | Decode | tok/step | ms/step | Prefill 7.8K | Prefill 31K | KV pool | GPUs |
|---|---|---|---|---|---|---|---|
| **MTP, `num_speculative_tokens=3`** (baseline) | 136.9 tok/s | 3.13 | 22.87 | 7078 tok/s | 6396 tok/s | 646,432 tok (2.47x) | 1 |
| **DFlash2, `num_speculative_tokens=7`, TP=1** | **249.7 tok/s (+82%)** | 5.12 | 20.50 | — | — | 521,044 (1.99x) | 1 |
| **DFlash2, TP=2** | **316.3 tok/s (+131%)** | 5.00 | 15.79 | 8260 | 8313 | 1,292,878 (4.93x) | 2 |

- Run-to-run spread is **0.1 tok/s** — reproducible, not lucky.
- TP=2 also gives **+30% prefill** (8313 vs 6396 at 31K) and **doubles** the KV pool that DFlash2 TP=1 had cut.
- Day-one number for the same config, **before** the context-KV fix was properly in place: **204.8 tok/s (+49.6%)**. Both are real measurements from the same box at different patch states.

**Why TP=2 speeds up a step that is latency-bound over PCIe without NVLink is not explained by us.** Step time drops 20.50 → 15.79 ms while accepted tok/step drops slightly (5.12 → 5.00). Reproducible across 3 configs (spread 0.1). Reported as an open question rather than a made-up mechanism.

**Sanity checks in the DFlash2 config (all live, on :8000)**

| Check | Result |
|---|---|
| reasoning split from content (`--reasoning-parser qwen3`) | `reasoning_len=124`, `content='\n\nhola'` OK |
| tool calling (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`) | `get_weather {"ciudad":"Lima"}` OK |
| **vision** | image → `'Azul'` OK (drafter logs `does not support external multimodal embeddings` — **inocuous**, the target's tower still runs) |
| structured JSON extraction | 4/4 fields OK |

## 4. The serve command (exactly what we run)

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0     # flashinfer JIT fails in our env (code=127); drafters don't need it
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_CPU_OMP_THREADS_BIND=close

vllm serve /path/to/qwen38-27b-gptq-int4 \
  --served-model-name Qwen3.8-27B-VL-MTP-Tools \
  --port 8000 --host 0.0.0.0 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --disable-log-stats \
  --speculative-config '{"method": "dflash", "model": "/path/to/qwen38-dflash2-w4a16", "num_speculative_tokens": 7}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

For 2 GPUs: `--tensor-parallel-size 2` — that is the whole diff (both GPUs must be visible to the job).

Two flags are easy to get wrong, and both change the number you measure:
- **`--mamba-ssm-cache-dtype float16`** — without it you get ~118 tok/s instead of ~137 for the *same* config. It is not a drafter thing; the MTP baseline needs it too. If your baseline looks ~15% low, check this first.
- **`--speculative-config num_speculative_tokens`** — swept for MTP: 1→84.6, 2→114.5, **3→118.3–137**, 8→114.2 tok/s (3 is the MTP optimum). We use **7** for DFlash2.

## 5. The patch you need (the actual contribution)

**Symptom.** With a *quantized* drafter, vLLM dies at load with:

```
AttributeError: 'QKVParallelLinear' object has no attribute 'weight'
```

**Root cause — it is the checkpoint, not vLLM's drafter support.** The W4A16 checkpoint stores q/k/v as **separate tensors** (`self_attn.q_proj.weight_packed`, `k_proj`, `v_proj`) instead of a fused `qkv_proj.weight`. `QKVParallelLinear` therefore never builds a dense `.weight` and only exposes `weight_packed` / `weight_scale` / `weight_shape`. But `_build_context_kv_buffers()` precomputes the context-KV projection at load time with a line that **assumes fusion**:

```python
kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
```

**Unfixed upstream in both 0.28.0 and 0.29.0.** We diffed the two tags file-by-file: `_build_context_kv_buffers` is **byte-identical** (empty diff), only moved from line 490 → 472. Upgrading vLLM does **not** fix it.

**Fix** (`dflash2-quantized-drafter-kv.patch` — 39 added lines, 1 changed, against `vllm/model_executor/models/qwen3_dflash.py`):

```python
def _dense_kv_rows(attn: nn.Module) -> torch.Tensor:
    """Return the K/V rows of a QKV projection as a dense matrix.

    The W4A16 DFlash2 checkpoint uses compressed-tensors for qkv_proj. In
    this vLLM version that layer exposes packed weights rather than ``weight``;
    context-KV precomputation still needs the dense K/V rows.
    """
    qkv = attn.qkv_proj
    weight = getattr(qkv, "weight", None)
    if weight is not None and weight.dim() == 2:
        return weight[attn.q_size :]

    packed, scale = qkv.weight_packed, qkv.weight_scale
    out_features, in_features = int(packed.shape[0]), int(qkv.input_size)
    bits = 32 * packed.shape[1] // in_features
    from compressed_tensors.compressors.pack_quantized.base import unpack_from_int32

    quantized = unpack_from_int32(
        packed.data, bits, torch.Size([out_features, in_features]), packed_dim=1,
    )
    group_size = in_features // scale.shape[1]
    dense = (
        quantized.to(torch.float32)
        .reshape(out_features, in_features // group_size, group_size)
        * scale.to(torch.float32)[..., None]
    ).reshape(out_features, in_features)
    dtype = scale.dtype if scale.dtype.is_floating_point else torch.bfloat16
    return dense.to(dtype)[attn.q_size :]
```

and the one-line call-site change:

```diff
-        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
+        kv_weights = [_dense_kv_rows(a) for a in layers_attn]
```

**Apply it** (from inside `site-packages/`):

```bash
cd "$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/.."
cp vllm/model_executor/models/qwen3_dflash.py{,.bak}
patch -p1 --dry-run < dflash2-quantized-drafter-kv.patch   # confirm it applies first
patch -p1            < dflash2-quantized-drafter-kv.patch
```

We verified it end to end: applied to a **pristine 0.28.0 copy** of the file — `--dry-run` clean, real apply, then `ast.parse()` on the result plus an assertion that the call site carries the helper.
SHA-256 of the patch: `a13c444641e6a7e5cf39dc3355cf3a5f3c208beb4e4328ce5620d4791a236c47`

> **Persistence warning:** this is a **hand patch inside `site-packages`**. Any `pip install -U vllm` silently overwrites the file and DFlash2 breaks again with the same `AttributeError`. Keep the `.patch` in version control and **re-apply it after every upgrade, before restarting**. Upstream-able: the helper is generic and only activates when `.weight` is absent — it cannot change behaviour for dense drafters.

## 6. Honest caveats

1. **DFlash2 TP=1 costs you KV cache.** 646,432 → **521,044 tokens** (2.47x → 1.99x of a 262K request), plus **slower startup** (580 s to ready vs ~200–300 s). TP=2 fixes both (1,292,878) but occupies both GPUs.
2. **A decode speedup may not be what your workload needs.** On our live traffic (236 agent-style requests):
   - prompt tokens: **p50 74,974 · p90 172,013 · max 195,182**
   - completion tokens: p50 461 · p90 1651 · max 79,231

   → **98.6% of tokens are prefill**, and ~42% of requests exceed 80K prompt. At 172K prompt, prefill is ~85% of wall time. Drafters **only accelerate decode**; the lever that moves prefill is TP=2 (+30% measured). Measure your own prompt/completion split before chasing tok/s.
3. **The drafter's `does not support external multimodal embeddings` log line is noise** — vision keeps working (verified end to end).
4. **Warm up before measuring.** The first requests after startup lie until CUDA graphs are captured. We discard 2 warmup calls and take the median of 5.
5. **Never measure two servers at once.** Two concurrent queues contaminated runs (81.9 and 111.6 tok/s from the same config). Serialize. And when swapping configs on a fixed port, **check the engine actually started** (`Application startup complete` in the job log) — a crashed engine leaves the *previous* server answering on that port, and your "A/B" silently measures the old config.
6. **`--kv-cache-dtype fp8` is not free**: the pool nearly doubles, but it degraded all three of our real workloads and raised perplexity **38.28 vs 34.46 (+11%)**. Dropped.
7. Our GPUs are **PCIe with no NVLink**. H100 SXM / NVLink-connected results will differ (likely better for TP=2).

## 7. What did NOT help (measured, so you don't repeat it)

| Lever | Result |
|---|---|
| `VLLM_USE_RUST_FRONTEND=1` | 134.2 vs 136.9 tok/s — **no gain**; also logs `enable_auto_tool_choice currently has no effect in Rust frontend` (tool calling still worked) |
| `--gdn-prefill-backend triton` | 137.5 vs 136.9 tok/s — **noise** |
| `--tool-call-parser qwen3_xml` | identical to `qwen3_coder` — both resolve to the same parser (`qwen3_engine_tool_parser`) |
| `--max-num-batched-tokens 2048` | 125.2 tok/s — worse |
| `enable_thinking:false` / `reasoning_effort:low` | **more** tokens (575/538 vs 425), no saving on this abliterated variant |
| `--kv-cache-dtype fp8` | PPL +11%, see above |
| `--language-model-only` | saves VRAM, **breaks vision** |
| NVFP4 | needs fp4 kernels via flashinfer, and flashinfer JIT fails in our env; INT4 GPTQ is already W4A16 |
| vLLM 0.29.0 | **does not fix the drafter bug** (verified by diffing the tags); upgrading also wipes the patch |

## 8. Credits

- **DFlash2 drafter for Qwen3.8-27B:** `incoai/Qwen3.8-27B-DFlash2` (bf16 original) and `syv-ai/HyperQwen` (W4A16 repack that makes it fit on a 24 GB card, and the reference implementation of the lookup-drafting patch we adapted).
- **Target model lineage:** `llmfan46` (MTP-preserving uncensored merge) → GPTQ repack.
- **vLLM** for the block-drafter / DFlash2 serving path.

## 9. License / attribution

- **This repository** (report, benchmark script, patch): **Apache-2.0**.
- `dflash2-quantized-drafter-kv.patch` is an adaptation of the lookup-drafting work in
  [`syv-ai/HyperQwen`](https://github.com/syv-ai/HyperQwen) (Apache-2.0). Same licence, attribution kept.
- **Drafter model** `incoai/Qwen3.8-27B-DFlash2` — Apache-2.0. The W4A16 repack referenced here is the
  community requantization built for the same 24 GB-card effort.
- **Target model** lineage: `llmfan46/Qwen3.8-27B-Ultra-Uncensored-Heretic-Native-MTP-Preserved` (Apache-2.0) → INT4 GPTQ repack.
- **vLLM** is Apache-2.0; this patch is offered as a fix that is safe to upstream (the helper only activates
  when the dense `.weight` is absent, so dense drafters are unaffected).
- Numbers in this report were measured by us on 2x H100 PCIe; they are provided as-is, no warranty.
