# The Match Maker — On-Device LLM Matchmaking Game

A browser-based **AI matchmaking game** where **Gemma 4 runs entirely on your device** via MediaPipe Tasks GenAI (WebGPU). The large language model evaluates personality compatibility and generates match reasoning locally — no API key, no cloud inference, full privacy. The model is an **opt-in download**: nothing is fetched until you click **Download & Play**, and it's cached in your browser (OPFS) so later visits skip the download.

**[▶ Live Demo](https://woodyhoko.github.io/match-maker-game)**

---

## 1. What makes this technically interesting

Running a multi-billion-parameter LLM in a browser tab is a relatively recent capability, enabled by three converging advances:

1. **WebAssembly (WASM)** — near-native execution of compiled C++ inference kernels in the browser sandbox
2. **MediaPipe Tasks GenAI** — Google's inference runtime compiled to WASM, with support for LiteRT (TFLite) and gguf-format models
3. **Gemma 4 E2B** — a compact instruction-tuned model, distributed here as the token-free [`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) LiteRT build (the same model used by [portfolio_2026](https://github.com/woodyhoko/portfolio_2026))

The result: a one-time model download (≈ 2 GB), then full LLM inference in the browser with no server round-trips.

---

## 2. On-device LLMs in the browser

The Gemma family is built for on-device use, and the general techniques that make a multi-billion-parameter model fit in a browser tab are:

- **Low-bit quantization** — weights are stored in 4-bit integers (INT4) while activations are computed at higher precision, cutting the memory footprint several-fold
- **WebGPU acceleration** — inference kernels run on the GPU through the browser's WebGPU backend rather than the CPU
- **Streaming + OPFS caching** — the weights are streamed once and persisted in the Origin Private File System, so they don't have to be re-downloaded on the next visit

These let a model run within the ~6–8 GB VRAM budget of typical consumer GPUs. This project uses the **Gemma 4 E2B** LiteRT build; for exact, verified architecture details refer to Google's official Gemma documentation rather than this README.

---

## 3. MediaPipe Tasks GenAI integration

```javascript
import { FilesetResolver, LlmInference } from '@mediapipe/tasks-genai';

const fileset = await FilesetResolver.forGenAiTasks(
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm'
);

// Weights are streamed from the HF mirror and tee'd into the OPFS cache on first run.
const tracked = streamWithProgress(modelStream, totalBytes, onProgress);

llmInference = await LlmInference.createFromOptions(fileset, {
  baseOptions: { modelAssetBuffer: tracked.getReader() },
  maxTokens: 1024,
  topK: 8,           // sample from the 8 most likely tokens
  temperature: 0.8
});

const result = await llmInference.generateResponse(prompt);
```

> **Note on `topK`:** This game uses `topK: 8` — at each step the model samples from the 8 most likely next tokens (combined with `temperature: 0.8`). That keeps the AI clients' dialogue varied and natural while staying coherent, instead of the rigid, repetitive output you get from greedy `topK: 1` decoding. Lower it toward `1` for more deterministic scoring, or raise it (e.g. `40`) for more variety.

The model is fetched once from the token-free [`litert-community`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) mirror and cached in the **Origin Private File System (OPFS)** — subsequent loads read straight from disk with no re-download.

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
| On-device LLM | [Gemma 4 E2B](https://ai.google.dev/gemma) (INT4 LiteRT build) |
| Inference runtime | MediaPipe Tasks GenAI (WASM + WebGPU) |
| Model hosting | [litert-community on HuggingFace](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) (token-free) |
| Model cache | Origin Private File System (OPFS) |
| Styling | Custom CSS — glassmorphism, Outfit + Playfair Display fonts |
| Build | None — single HTML file, ES module imports |

---

## 8. Run locally

```bash
# ES modules + SharedArrayBuffer require a server with COOP/COEP headers
python -m http.server 8000
# open http://localhost:8000
# First load: one-time ~2 GB model download (opt-in; cached in OPFS after)
```

> `SharedArrayBuffer` requires `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers. The live demo host configures these; a plain `python -m http.server` will not. Use a configured dev server (e.g. `vite` with `server.headers`) for local development.
