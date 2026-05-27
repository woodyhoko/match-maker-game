# The Match Maker — Gemma 3n Edition

A browser-based **AI matchmaking game** powered by **Gemma 3n running on-device** via MediaPipe Tasks. The AI evaluates personality compatibility entirely in your browser — no API key, no cloud inference.

**[▶ Live Demo](https://woodyhoko.github.io/match-maker-game)**

---

## How It Works

1. Players fill out a personality profile
2. Gemma 3n (loaded from HuggingFace Hub) runs locally in WebAssembly
3. The model evaluates compatibility between profiles and generates a match score + reasoning
4. Results are displayed with a romantic flair ✨

---

## Tech Stack

| Layer | Technology |
|---|---|
| On-device LLM | [Gemma 3n](https://ai.google.dev/gemma) via MediaPipe Tasks GenAI |
| Model Hub | HuggingFace Hub (ESM) |
| Styling | Custom CSS with glassmorphism effects |
| Fonts | Outfit + Playfair Display |
| Build | None — single HTML file |

---

## Privacy

All inference happens **locally in your browser**. No text you enter is sent to any server. The Gemma 3n weights are downloaded once from HuggingFace and cached by the browser.

---

## Run Locally

```bash
# Serve (required — ES modules need a server context)
python -m http.server 8000
# then open http://localhost:8000
```

> First load downloads the Gemma 3n model weights (~1.5 GB). Subsequent loads use the browser cache.

