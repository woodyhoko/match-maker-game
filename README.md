# The Match Maker — On-Device LLM Matchmaking Game

A browser-based **AI matchmaking game** where **Gemma 3n runs entirely on your device** via MediaPipe Tasks GenAI. The large language model evaluates personality compatibility and generates match reasoning locally in WebAssembly — no API key, no cloud inference, full privacy.

**[▶ Live Demo](https://woodyhoko.github.io/match-maker-game)**

---

## 1. What makes this technically interesting

Running a multi-billion-parameter LLM in a browser tab is a relatively recent capability, enabled by three converging advances:

1. **WebAssembly (WASM)** — near-native execution of compiled C++ inference kernels in the browser sandbox
2. **MediaPipe Tasks GenAI** — Google's inference runtime compiled to WASM, with support for LiteRT (TFLite) and gguf-format models
3. **Gemma 3n** — a compact architecture (1B–4B parameters) specifically designed for on-device inference efficiency via the **MobileNet Embedding (MNE)** approach, which uses per-layer weight matrices shared across a multi-scale feature hierarchy

The result: a ~1.5 GB model download, then full LLM inference at 5–15 tokens/sec in the browser with no server round-trips.

---

## 2. Gemma 3n and on-device LLMs

**Gemma 3n** (released May 2025) is designed for on-device use, introducing:

- **MatMul-free attention** — reduces memory bandwidth for KV-cache by compressing key/value projections
- **Selective layer activation** — not all layers are evaluated for every token, reducing FLOP count
- **INT4/INT8 quantization** — model weights stored in 4-bit integers; activations computed in float16/bfloat16

These techniques allow a model with the capability of a larger LLM to run with the memory footprint of a smaller one — critical for the ~6–8 GB VRAM limit of typical consumer GPUs, and for browser WASM which shares system RAM.

---

## 3. MediaPipe Tasks GenAI integration

```javascript
import { FilesetResolver, LlmInference } from '@mediapipe/tasks-genai';

const fileset = await FilesetResolver.forGenAiTasks(
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm'
);

// Model binary streamed from OPFS cache (downloaded from HuggingFace on first run)
const assetReader = modelFile.stream().getReader();

llmInference = await LlmInference.createFromOptions(fileset, {
  baseOptions: { modelAssetBuffer: assetReader },
  maxTokens: 1024,
  topK: 1,           // source default — see note below
  temperature: 0.8
});

const result = await llmInference.generateResponse(prompt);
```

> **Note on `topK`:** The source code uses `topK: 1`, which is greedy decoding — always picking the single most likely next token. This makes match scores deterministic (same profile pair → same score every time), but it limits response variety in the conversational phase. `temperature: 0.8` has no effect when `topK=1`. For more natural, varied AI client dialogue, change `topK` to `40`; for purely deterministic scoring, keep `topK: 1`.

The model binary is downloaded once from HuggingFace Hub (ESM CDN) and cached by the browser's Cache Storage API — subsequent loads are instant.

---

## 4. Prompt design

The matchmaking prompt is a **few-shot structured prompt** that formats two personality profiles as JSON and asks Gemma to produce a compatibility score plus explanation:

```
You are a thoughtful matchmaker. Given two personality profiles, assess their
romantic compatibility on a scale of 0–100 and explain your reasoning.

Profile A: { "name": "...", "interests": [...], "values": [...], "personality": "..." }
Profile B: { "name": "...", "interests": [...], "values": [...], "personality": "..." }

Respond with:
SCORE: <0-100>
REASON: <2-3 sentences>
```

Structured output is extracted with a regex on the response rather than relying on JSON mode (not available in the MediaPipe Tasks API at time of writing).

---

## 5. Privacy model

| Data | Where it goes |
|---|---|
| Personality profile inputs | Stays in browser memory only |
| LLM inference | Runs in the WASM sandbox on-device |
| Model weights | Downloaded from HuggingFace; cached locally by the browser |
| Match results | Never transmitted anywhere |

There is **no telemetry, no logging, no server** involved after the model download.

---

## 6. Performance characteristics

| Hardware | Tokens/sec (approx.) |
|---|---|
| M2 MacBook Air (WebGPU via Chrome) | 15–25 |
| Desktop RTX 3080 (WebGPU) | 20–35 |
| Mobile mid-range (WASM CPU) | 2–5 |
| Desktop CPU-only WASM | 5–12 |

WebGPU acceleration (when available) is automatically used by MediaPipe; the WASM CPU path is the fallback.

---

## 7. Stack

| Layer | Technology |
|---|---|
| On-device LLM | [Gemma 3n](https://ai.google.dev/gemma) (INT4 quantized) |
| Inference runtime | MediaPipe Tasks GenAI (WASM + optional WebGPU) |
| Model hosting | HuggingFace Hub (ESM CDN) |
| Styling | Custom CSS — glassmorphism, Outfit + Playfair Display fonts |
| Build | None — single HTML file, ES module imports |

---

## 8. Run locally

```bash
# ES modules + SharedArrayBuffer require a server with COOP/COEP headers
python -m http.server 8000
# open http://localhost:8000
# First load: ~1.5 GB model download (cached after)
```

> `SharedArrayBuffer` requires `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers. The live demo host configures these; a plain `python -m http.server` will not. Use a configured dev server (e.g. `vite` with `server.headers`) for local development.
