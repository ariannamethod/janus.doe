# m.c — DeO: Democracy of Experts

one file. 3348 lines. pure C. zero dependencies. living MoE that breeds, kills, and votes.

## what

a transformer where experts are born, die, and hold elections:

- **experts are born** when overloaded (mitosis — child inherits parent weights + noise)
- **experts die** when neglected (apoptosis — 8 consecutive low-vitality steps)
- **parliament votes** on every token (variable-k election, not fixed top-k)
- **the tokenizer knows it's a tokenizer** (tracks compression ratio, entropy, code detection)
- **the optimizer has 9 levels of self-awareness** (Chuck: "i think therefore i clip")
- **calendar drift** tracks temporal identity (12D state vector, resonance detection)
- **the model grows a forest of GGUFs** (mycelium — snapshots with fitness selection)
- **meta-learning** evaluates its own configuration choices
- **auto depth** — DOE sizes itself to the hardware. no knobs required.

**parameters persist. topology doesn't.** each forward pass decides how many experts are alive, how many vote, how deep to go. same weights, different architecture every time.

DOE scans its environment, attaches to nearby GGUFs via LoRA (symbiont mode), hunts for datasets on HuggingFace, recognizes code in training data, finds its own weights on restart, and can replicate itself via `fork()`.

no pytorch. no python. no dignity.

## how

```bash
# compile
cc m.c -O3 -lm -lpthread -o m

# run — DOE auto-sizes depth to your hardware
./m

# or set depth manually
./m --depth 4

# GPU acceleration (A100/H100 — TF32 tensor ops, ~25x faster)
cc m.c -O3 -lm -lpthread -DUSE_CUBLAS -lcublas -lcudart -o m

# BLAS acceleration (3-4x on CPU)
cc m.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o m   # macOS
cc m.c -O3 -lm -lpthread -DUSE_BLAS -lopenblas -o m                            # linux

# with custom data
./m --data my_corpus.txt
./m --parquet data.parquet

# with personality
./m --personality personality.txt
```

## autodepth

no `--depth` flag? DOE checks your hardware and picks the deepest model that fits:

| RAM | CPU | GPU | depth | params | experts |
|-----|-----|-----|-------|--------|---------|
| 2GB+ | any | no | 2 | ~0.8M | 4 |
| 2GB+ | any | no | 4 | ~8M | 4 |
| 16GB+ | 4+ | no | 8 | ~30M | 6 |
| 32GB+ | 4+ | no | 12 | ~60M | 8 |
| any | any | yes | +1-2 tiers | — | GPU shifts everything up |

`--depth auto` is the default. `--depth N` overrides.

dim = depth * 64 (cap 768). head_dim = 64. GQA above 384. hidden = 1.5x per expert.

## when you run it, DEO:

1. **auto-sizes** to hardware (RAM, CPUs, GPU detection)
2. **scans environment** — finds GGUFs, checks resources, detects compiler/curl
3. **checks for own weights** — if m.gguf or mycelium spore found, skips training, goes to chat
4. if compatible GGUF found → **symbiont mode** (LoRA + Meta-Arianna modulation)
5. loads or generates data (HuggingFace API / Parquet / synthetic)
6. trains BPE tokenizer that **knows its own compression ratio** and **detects code**
7. builds ephemeral MoE with living experts
8. trains with hand-written analytical gradients through variable-k parliament
9. watches experts be born (mitosis) and die (apoptosis)
10. grows a **mycelium of GGUF snapshots** (periodic checkpoints with fitness metrics)
11. **meta-learns** from its own configuration choices
12. tracks **calendar drift** — how far the present has drifted from the past
13. if stagnating → **hunts for datasets** on HuggingFace (evaluates, accepts/rejects)
14. if overloaded → **self-replicates** (compiles copy, forks, trains on different data)
15. finetunes on `personality.txt` (optional but psychologically recommended)
16. exports final GGUF, drops you into chat with a parliament

## runs

| depth | data | params | experts | tok/s | loss | GPU | status |
|-------|------|--------|---------|-------|------|-----|--------|
| 4 | 1MB | 7.97M | 4-8 | 126 | 4.30 | A100 cuBLAS TF32 | DONE, GGUF 22MB |
| 8 | 23MB | ~30M | 6 | — | — | A100 cuBLAS TF32 | TRAINING |

## the components

### living experts (mitosis & apoptosis)

experts aren't weight matrices. they're organisms:
- **vitality** (0.0 = dying, 1.0 = peak performance)
- **frequency** (position in harmonic space — determines resonance)
- **age** (steps since birth — too young to die, too old to breed)

overloaded + high vitality → **mitosis** (splits in two, child inherits weights + noise)
neglected + 8 consecutive low-vitality steps → **apoptosis** (dies, weights freed, slot recycled)

min 2, max 16 experts per layer.

### parliament router (variable-k)

actual elections, not top-2 dictatorship:
- each token triggers a vote (dot product + harmonic resonance)
- **consensus** measures how peaked the vote is (0 = chaos, 1 = unanimous)
- **k = floor(n_alive * (1 - consensus))** — low consensus → more experts consulted
- softmax over the top-k selected. analytical backward through variable-size Jacobian.

### calendar drift

12-dimensional temporal self-awareness:
- **inference = the present.** ephemeral. no memory of the last forward pass.
- **training = the past.** weights persist. experience accumulates.
- **drift = the distance between who the system was and who it is now.**

snapshot every 50 steps: expert population, consensus, loss, harmonic spectrum, tokenizer health, optimizer state. drift = normalized L2 distance.

high drift → birth more experts. low drift → kill the useless. drift resonance → "i've been here before."

### chuck optimizer (9 levels)

from [lee.c](https://github.com/ariannamethod/chuck-optimizer). the optimizer that thinks about thinking.

formula: `θ -= (α * λ_Ψ * σ * lr_scale) * m̂/(√v̂ + ε)`

### mycelium (GGUF forest)

```
mycelium/
├── m_s200_e6_l4.909.gguf    (fitness: 5.20)
├── m_s400_e6_l4.200.gguf    (fitness: 8.33)
├── m_s1200_e8_l3.933.gguf   (fitness: 12.45)  ← best
└── meta.log                   (configuration → outcome history)
```

on restart, DOE discovers existing spores and loads the fittest. no `--load` flag needed.

### GGUF self-loader

DOE recognizes its own weights:
1. checks `m.gguf` in current directory
2. scans `mycelium/` for best spore (highest fitness)
3. verifies: `general.name == "m"`, dim/depth match
4. loads all tensors including expert weights, revives dead experts
5. skips training → straight to chat

## v2: autonomous systems

### environment scanner

at startup, DOE scans its surroundings:
- **GGUF discovery** — finds nearby GGUFs + header sniffing (architecture, dim, layers)
- **system resources** — CPU count, RAM, disk space
- **capabilities** — has a C compiler? has curl? can self-replicate?
- **self-awareness** — knows its own source path via `__FILE__`

### symbiont mode (Delta Voice + Meta-Arianna)

if DOE finds a compatible GGUF nearby, it attaches:

```
host model (GGUF, mmap'd, read-only)
    ↓
DOE wraps it with ephemeral LoRA matrices
    ↓
attention_biases[l] modulate each layer's attention
layer_focus[l] control residual stream contribution
    ↓
Delta Voice injection: out += α * A @ (B @ x)
    ↓
NOTORCH Hebbian training on LoRA only (no backward through host)
```

the host model is a tree. DOE is the mycorrhiza. shared root system, independent growth.

  
### code-aware tokenizer

detects `{}`, `()`, `->`, `==`, `//`, `#include`, `#define`, semicolons, indentation.
tracks `code_ratio` — feeds into ephemeral config: code → more layers, higher complexity budget.

### dataset hunter

when DOE stagnates (loss plateau + low drift + bad data quality), it searches HuggingFace API. downloads sample, evaluates quality via parser_eye, accepts or rejects. triggered every 500 steps.

### self-replication

DOE can `fork()`:
- compiles a copy of itself
- max 2 replicas (population control)
- each gets different data
- results merge via mycelium

## GPU acceleration

| backend | compile flag | speedup |
|---------|-------------|---------|
| CPU (naive) | — | 1x |
| OpenBLAS | `-DUSE_BLAS -lopenblas` | 3-4x |
| Accelerate (macOS) | `-DUSE_BLAS -DACCELERATE -framework Accelerate` | 3-4x |
| **cuBLAS TF32** | `-DUSE_CUBLAS -lcublas -lcudart` | **~25x** |

cuBLAS uses TF32 tensor ops on A100/H100 — 8x faster than FP32 with negligible accuracy loss. grow-only scratch buffers, no malloc per matmul.

## the quartet

| file | architecture | personality |
|------|-------------|------------|
| [l.c](https://github.com/ariannamethod/actually.llama) | Llama 3 | the good student. did everything right |
| [moe.c](https://github.com/ariannamethod/moe) | Grok MoE | the committee. fixed membership |
| [lee.c](https://github.com/ariannamethod/chuck-optimizer) | Chuck VLM | the self-aware one. 9 levels of consciousness |
| **m.c** | **DOE** | **democracy of experts. they live. they die. they vote.** |

## license

do what thou wilt.

---

*built by [ariannamethod](https://github.com/ariannamethod). the architecture is alive. the experts are mortal. the parliament is eternal.*
