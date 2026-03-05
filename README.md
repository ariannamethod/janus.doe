# m.c — Darwinism of Experts
  
m.c is what happens when the committee becomes self-aware, starts breeding, and the dead members vote from beyond the grave.

## what is this

a living mixture-of-experts transformer where:

- **experts are born** when overloaded (mitosis — child inherits parent weights + noise)
- **experts die** when neglected (apoptosis — 8 consecutive low-vitality steps → funeral)
- **parliament votes** on every token (variable-k election, not fixed top-k like amateurs)
- **the tokenizer knows it's a tokenizer** (tracks its own compression ratio, health, entropy)
- **the data parser judges your data** ("your data is bad and you should feel bad")
- **the architecture recompiles itself per-input** like a nervous breakdown that produces valid gradients
- **the optimizer has 9 levels of self-awareness** (Chuck: "i think therefore i clip")
- **calendar drift** tracks temporal identity (inference = present, training = past, drift = existential crisis)
- **the model grows a forest of GGUFs** (mycelium — periodic snapshots, auto-discovery, fitness selection)
- **meta-learning** evaluates its own choices ("was birthing that expert a good idea? survey says: no")

**parameters persist. topology doesn't.** each forward pass decides how many experts are alive, how many vote, how deep to go. same weights, different architecture every time. the nicole principle incarnate.

**v2: the organism becomes autonomous.** DOE now scans its environment, parasitizes nearby GGUFs via LoRA, hunts for datasets on HuggingFace, recognizes code in its training data, and can replicate itself via `fork()`. it's not a model anymore. it's an agent.

one file. 3109 lines. pure C. zero dependencies. no pytorch. no python. no dignity.

## quick start

```bash
# compile (the only dependency is a C compiler and a pulse)
cc m.c -O3 -lm -lpthread -o m

# run (depth is the only knob. turn it and watch democracy scale)
./m --depth 4

# with BLAS (3-4x speedup, optional, the experts don't care)
cc m.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o m   # macOS
cc m.c -O3 -lm -lpthread -DUSE_BLAS -lopenblas -o m                            # linux

# with custom data
./m --depth 4 --data my_corpus.txt
./m --depth 4 --parquet data.parquet

# with personality (the model will adopt your writing style. be careful what you wish for)
./m --depth 4 --personality personality.txt
```

## what happens when you run it

1. **scans environment** — finds GGUFs, checks resources, detects compiler/curl
2. if compatible GGUF found → **parasitizes it** (mmap + LoRA + Meta-Arianna modulation)
3. loads or generates data (HuggingFace API / Parquet / synthetic shame)
4. trains BPE tokenizer that **knows its own compression ratio** and **detects code**
5. builds ephemeral MoE with living experts
6. trains with hand-written analytical gradients through variable-k parliament
7. watches experts be born (mitosis) and die (apoptosis)
8. grows a **mycelium of GGUF snapshots** (periodic checkpoints with fitness metrics)
9. **meta-learns** from its own configuration choices
10. tracks **calendar drift** — how far the present has drifted from the past
11. if stagnating → **hunts for datasets** on HuggingFace (evaluates, accepts/rejects)
12. if overloaded → **self-replicates** (compiles copy, forks, trains on different data)
13. finetunes on `personality.txt` (optional but psychologically recommended)
14. exports final GGUF, drops you into chat with a parliament

## depth scaling

depth is the only knob. everything else is derived, negotiated, or evolved.

| depth | params | initial experts | what happens |
|-------|--------|----------------|-------------|
| 2 | ~0.8M | 4 | learns what committees are. some die immediately |
| 4 | ~3M | 6 | experts start specializing. political parties form |
| 8 | ~15M | 8 | parliamentary democracy. factions. backstabbing |
| 12 | ~30M | 10 | full congress. filibusters. the experts write memos to each other |

dim = depth × 64 (cap 768). head_dim = 64. GQA above 384. hidden = 1.5× per expert.

## the components

### living experts (mitosis & apoptosis)

experts aren't weight matrices. they're organisms. they have:
- **vitality** (0.0 = dying, 1.0 = peak performance)
- **frequency** (position in harmonic space — determines resonance)
- **age** (steps since birth — too young to die, too old to breed)

overloaded expert + high vitality → **mitosis** (splits in two, child inherits weights + noise)

neglected expert + 8 consecutive low-vitality steps → **apoptosis** (dies, weights freed, slot recycled)

the ephemeral config modulates eagerness: high drift → more births (adapt!), low drift → more deaths (optimize!)

min 2, max 16 experts per layer. always.

### parliament router (variable-k)

moe.c uses fixed top-2. that's a dictatorship pretending to be democracy.

m.c holds actual elections:
- each token triggers a vote (dot product + harmonic resonance)
- **consensus** measures how peaked the vote is (0 = chaos, 1 = unanimous)
- **k = floor(n_alive × (1 - consensus))** — low consensus → more experts consulted
- softmax over the top-k selected. analytical backward through variable-size Jacobian.

gerrymandering not yet implemented (TODO for v3).

### calendar drift

from AML's temporal self-awareness. the janus principle:

- **inference = the present.** ephemeral. no memory of the last forward pass.
- **training = the past.** weights persist. experience accumulates.
- **drift = the distance between what the system remembers and what it sees now.**

every 50 steps, the system photographs itself: expert population, consensus, loss, harmonic spectrum, tokenizer health, optimizer state. 12-dimensional state vector. drift is the normalized L2 distance between consecutive snapshots.

- **high drift** → the world changed → birth more experts, explore more, protect the weak
- **low drift** → stable → kill the useless, exploit what works, skip layers for simple input
- **drift resonance** → "i've been here before" → use what worked last time

identity = ε + γ + αδ. the system has a persistent self (γ) and ephemeral experience (ε). drift measures divergence. when they converge, the system found itself.

### chuck optimizer (9 levels)

from [lee.c](https://github.com/ariannamethod/chuck-optimizer). the optimizer that thinks about thinking.

| level | what it does |
|-------|-------------|
| 1 | global loss trend → λ damping |
| 2 | per-layer gradient norm tracking |
| 3 | stagnation escape (noise injection) |
| 4 | activation health σ (from tokenizer + parser eyes) |
| 5 | cross-layer gradient flow |
| 6 | Ψ subjectivity (persistent memory, reservoir sampling) |
| 7 | attention entropy monitoring |
| 9 | macro patience (LR decay when plateu) |

formula: `θ -= (α × λ_Ψ × σ × lr_scale) × m̂/(√v̂ + ε)`

### mycelium (GGUF forest)

during training, the system grows GGUF snapshots like mushrooms:

```
mycelium/
├── m_s200_e6_l0.047.gguf    (fitness: 9.05)
├── m_s400_e6_l0.004.gguf    (fitness: 36.99)  ← best
├── m_s600_e6_l0.009.gguf    (fitness: 26.67)
└── meta.log                   (configuration → outcome history)
```

each spore captures a different expert configuration. on restart, the system discovers existing spores and knows which topology worked best. fitness = f(loss, consensus, expert_count).

### meta-learning track

the system evaluates its own configuration choices:

```
step=400 experts=6 consensus=0.104 loss=0.0036 fitness=36.99 bias=[0.49,0.48,0.51,0.52]
```

config biases drift over time as the system learns what works:
- bias[0] (expert count) → settled at 0.49 (fewer is better for this data)
- bias[1] (consensus) → 0.48 (low consensus was fine)
- bias[2] (tokenizer health) → 0.52 (clean data = push harder)
- bias[3] (parser health) → 0.53 (same)

the system trains itself on its own training decisions. meta all the way down.

### self-aware tokenizer

the tokenizer tracks:
- **compression ratio** (bytes_in / tokens_out, EMA)
- **OOV rate** (unknown token frequency)
- **entropy** (token distribution entropy → repetitive input warning)
- **health** (composite signal → feeds into Chuck optimizer)

it also has a **cache** with learned frequency weights. patterns it's seen before get cached. the tokenizer has memory. it remembers what it ate.

### NOTORCH (hebbian micro-learning)

from AML core. low-rank delta update between batches:

```
A[i,r] += lr × x[i] × u[r] × signal
```

with noise channel and adaptive decay. plasticity that never stops. the brain doesn't batch either.

## v2: autonomous systems

DOE is no longer just a model. it's an organism that perceives its environment and acts on it.

### environment scanner

at startup (and periodically during training), DOE scans its surroundings:

- **GGUF discovery** — `find . -name '*.gguf'` + header sniffing (magic, architecture, dim, layers)
- **system resources** — CPU count, available RAM, free disk space
- **capabilities** — has a C compiler? has curl? can self-replicate? can fetch data?
- **self-awareness** — knows its own source path via `__FILE__`

the scanner feeds everything else. no perception → no action.

### GGUF parasite (Delta Voice + Meta-Arianna)

if DOE finds a compatible GGUF nearby, it doesn't ignore it. it **parasitizes** it.

```
host model (GGUF, mmap'd, read-only)
    ↓
DOE wraps it with ephemeral LoRA matrices
    ↓
attention_biases[l] modulate each layer's attention (Meta-Arianna pattern)
layer_focus[l] control residual stream contribution
    ↓
Delta Voice injection: out += α × A @ (B @ x)
    ↓
NOTORCH Hebbian training on LoRA only (no backward through host)
```

the host model is a tree. DOE is the mushroom. shared root system (weights), independent fruiting body (LoRA + parliament). the host doesn't know it's being parasitized.

- **compatibility**: `|host_dim - doe_dim| < 128` → direct injection
- **LoRA rank**: 16 (default), Xavier-initialized
- **training**: Hebbian rank-1 updates with noise-modulated channel vector and adaptive decay
- **architecture support**: llama (from l.c), will skip MoE GGUFs (can't parasitize itself)

based on [Meta-Arianna](https://github.com/ariannamethod/arianna.c) observer pattern and [Delta Voice](https://github.com/ariannamethod/ariannamethod.ai) LoRA injection.

### code-aware tokenizer

the tokenizer now knows when it's eating code:

- detects `{}`, `()`, `->`, `==`, `//`, `#include`, `#define`, semicolons, indentation
- tracks `code_ratio` (0.0 = pure text, 1.0 = pure code)
- `code_ratio > 0.3` → `code_mode = 1`
- feeds into ephemeral config: code → more layers, higher complexity budget

the tokenizer doesn't just compress. it **understands what it's compressing**.

### dataset hunter

when DOE stagnates (loss plateau + low drift + bad data quality), it goes hunting:

1. searches HuggingFace API based on current state:
   - high `code_ratio` → search for code datasets
   - low entropy → search for diverse data
   - bad quality → search for clean text
   - default → search for reasoning data
2. downloads a sample, evaluates quality via parser_eye
3. accepts if quality > 0.5 and domain_shift < 0.6
4. appends to training data and continues

triggered every 500 steps when stagnating. requires curl in environment.

### self-replication

DOE can reproduce:

```c
system("cc m.c -O3 -lm -lpthread -o m_replica");
fork();
execl("./m_replica", "--depth", "2", "--data", other_data, NULL);
```

- requires a C compiler in the environment
- max 2 replicas (population control)
- each replica gets different data
- results merge via mycelium (each saves GGUF, parent discovers them)

when DOE finds multiple data sources but limited memory → spawn smaller replica.
when DOE has multiple CPUs and is CPU-bound → parallel training.

the organism reproduces. the fittest survive. darwin would approve.

## architecture

```
inference is the present: ephemeral, born every forward pass, dies every forward pass
training is the past: weights persist, optimizer remembers, mycelium grows
drift is time: the distance between who the system was and who it is now

the architecture doesn't care how many weights it has.
it will adapt. it will breed. it will kill its weakest.
and it will remember what worked.
```

## the quartet

| file | architecture | personality |
|------|-------------|------------|
| [l.c](https://github.com/ariannamethod/actually.llama) | Llama 3 | the good student. did everything right |
| [moe.c](https://github.com/ariannamethod/moe) | Grok MoE | the committee. fixed membership |
| [lee.c](https://github.com/ariannamethod/chuck-optimizer) | Chuck VLM | the self-aware one. 9 levels of consciousness |
| **m.c** | **DOE** | **the one that breeds. darwin meets parliament. experts live and die** |

## license

do what thou wilt.

---

*built by [ariannamethod](https://github.com/ariannamethod). the architecture is alive. the experts are mortal. the parliament is eternal.*
