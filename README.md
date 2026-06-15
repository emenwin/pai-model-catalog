# pai-model-catalog

Static model catalog for the PAI on-device AI runtime.

Clients fetch `catalog-index.json` as the entry point, then resolve each `listRef` relative to it to obtain per-engine model manifests.

[https://github.com/emenwin/pai-model-catalog](https://github.com/emenwin/pai-model-catalog)
---

## Repository layout

```
catalog-index.json          # Top-level index — one entry per module × engine
asr-mlx-models.json         # ASR manifests for the MLX engine
asr-whispercpp-models.json  # ASR manifests for the whisper.cpp engine
llm-shared-mlx-models.json  # Gemma shared LLM manifests for MLX
llm-chat-mlx-models.json    # Qwen chat-only manifests for MLX
```

---

## How it works

### Two-level structure

```
catalog-index.json
  └─ items[].listRef.url  →  ./asr-mlx-models.json
                          →  ./asr-whispercpp-models.json
                          →  ./llm-shared-mlx-models.json
                          →  ./llm-chat-mlx-models.json
                          →  ...
```

1. Client fetches `catalog-index.json` (supports `ETag` / `If-None-Match` caching).
2. Client filters `items` by `platform` + `moduleId` (+ optional `engine`).
3. For each matched item, client resolves `listRef.url` relative to the index URL and fetches the model list.
4. Client verifies the downloaded file against `listRef.sha256` (when non-empty).
5. Manifests are registered into the local model registry.
6. Shared Gemma manifests can then be projected to Translate, Chat, and Agent via `capabilities.supportedModules`.

Both index items and manifests expose `supportedModules`. This lets a Chat-only
client fetch the Qwen Chat list plus the shared Gemma list without fetching ASR
or unrelated model families.

### Resolving relative URLs

```swift
let indexURL = URL(string: "https://raw.githubusercontent.com/emenwin/pai-model-catalog/main/catalog-index.json")!
let listURL  = URL(string: "./asr-mlx-models.json", relativeTo: indexURL)!
// → https://raw.githubusercontent.com/<org>/pai-model-catalog/main/asr-mlx-models.json
```

---

## Schema

### `catalog-index.json`

| Field | Type | Description |
|---|---|---|
| `catalogVersion` | string | `YYYY.MM.DD-NNN` monotonically increasing |
| `generatedAt` | string | ISO 8601 UTC |
| `ttlSeconds` | number | Suggested client-side cache TTL |
| `items[].moduleId` | string | e.g. `module.asr` |
| `items[].supportedModules` | string[] | Optional module projections used while filtering the index |
| `items[].engine` | string | `mlx`, `whispercpp`, … |
| `items[].platforms` | string[] | `ios`, `macos` |
| `items[].listRef.url` | string | Relative URL to the model list file |
| `items[].listRef.sha256` | string | SHA-256 of the list file (empty = skip verification) |

### Model list files (e.g. `asr-mlx-models.json`)

Each file is a JSON array of manifest objects:

| Field | Type | Description |
|---|---|---|
| `modelId` | string | Unique model identifier |
| `moduleId` | string | Owning module |
| `version` | string | Semver |
| `platforms` | string[] | Supported platforms |
| `files[].path` | string | Relative path within the model bundle |
| `files[].role` | string | `weights`, `config`, `tokenizer`, … |
| `files[].sha256` | string | SHA-256 of the file |
| `files[].sizeBytes` | number | File size in bytes |
| `files[].downloadUrl` | string | Absolute download URL (Hugging Face, CDN, …) |
| `recommendedEngines` | string[] | Preferred engines in priority order |
| `backendKind` | string | Backend family such as `mlx`, `gemma-mlx`, `qwen-mlx` |
| `capabilities.supportedModules` | string[] | Functional projections, e.g. Translate + Chat |
| `capabilities.taskProfiles` | string[] | Supported task kinds |
| `capabilities.promptProfiles` | object | Prompt profile mapping used by runtime backends |
| `capabilities.supportsToolCalling` | boolean | Whether the model/chat template supports Agent tool calls |
| `constraints.minOs` | string | Minimum OS version |
| `constraints.minRamMB` | number | Minimum RAM in MB |
| `constraints.supportsStreaming` | boolean | Whether streaming inference is supported |
| `license` | string | SPDX identifier or `unknown` |

---

## Updating the catalog

1. Install generator dependencies with `python3 -m pip install -r Scripts/requirements.txt`.
2. Run `python3 Scripts/fetch_llm_model_catalog.py` for Gemma 4 and Qwen 3.5.
3. Run the corresponding ASR generator when updating Whisper catalogs.
4. Update `catalogVersion` and `generatedAt` in `catalog-index.json`.
5. Recompute `sha256` for changed lists when hash verification is enabled.

Catalog discovery uses `huggingface_hub.HfApi`. `transformers` is intentionally
not a catalog-search dependency: it is used by runtimes for model loading and
`apply_chat_template`, while Hub search, metadata, and file enumeration belong
to `huggingface_hub`.

---

## License

Catalog metadata is released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Individual model weights are subject to their respective licenses as noted in each manifest's `license` field.
