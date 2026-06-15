#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from huggingface_hub_client import HuggingFaceHubClient

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CHAT_OUTPUT = SCRIPT_DIR.parent / "llm-chat-mlx-models.json"
DEFAULT_SHARED_OUTPUT = SCRIPT_DIR.parent / "llm-shared-mlx-models.json"

RUNTIME_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


class HubClient(Protocol):
    def search_models(self, **kwargs: Any) -> list[dict[str, Any]]: ...
    def model_info(self, repo_id: str) -> dict[str, Any]: ...
    def file_url(self, repo_id: str, filename: str, revision: str = "main") -> str: ...


@dataclass(frozen=True)
class FamilySpec:
    name: str
    search: str
    repo_pattern: re.Pattern[str]
    module_id: str
    backend_kind: str
    supported_modules: tuple[str, ...]
    prompt_prefix: str
    min_ram_mb: int
    output: str


FAMILIES = (
    FamilySpec(
        name="qwen3.5",
        search="Qwen3.5",
        repo_pattern=re.compile(r"^Qwen3\.5-(?:0\.8B|2B|9B)-(?:4bit|8bit|bf16)$", re.IGNORECASE),
        module_id="module.llm.chat",
        backend_kind="qwen-mlx",
        supported_modules=("module.llm.chat", "module.agent"),
        prompt_prefix="qwen3.5",
        min_ram_mb=3072,
        output="chat",
    ),
    FamilySpec(
        name="gemma4",
        search="gemma-4",
        repo_pattern=re.compile(r"^gemma-4-e2b-it-(?:4bit|8bit|bf16)$", re.IGNORECASE),
        module_id="module.llm.shared",
        backend_kind="gemma-mlx",
        supported_modules=("module.translation", "module.llm.chat", "module.agent"),
        prompt_prefix="gemma4",
        min_ram_mb=5120,
        output="shared",
    ),
)


def selected_runtime_files(siblings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for sibling in siblings:
        filename = sibling.get("rfilename", "")
        is_weight = filename.endswith(".safetensors")
        if "/" in filename or (filename not in RUNTIME_FILES and not is_weight):
            continue
        selected.append(sibling)
    return sorted(selected, key=lambda item: item.get("rfilename", ""))


def file_role(filename: str) -> str:
    if filename.endswith(".safetensors"):
        return "weights"
    if filename in {"vocab.json"}:
        return "vocab"
    if filename == "merges.txt":
        return "merges"
    if "token" in filename or filename == "chat_template.jinja":
        return "tokenizer"
    return "config"


def build_manifest(model: dict[str, Any], spec: FamilySpec, client: HubClient) -> dict[str, Any] | None:
    repo_id = model.get("id", "")
    repo_name = repo_id.rsplit("/", 1)[-1]
    if not spec.repo_pattern.fullmatch(repo_name):
        return None

    runtime_files = selected_runtime_files(model.get("siblings", []))
    filenames = {item.get("rfilename", "") for item in runtime_files}
    if "config.json" not in filenames or not any(name.endswith(".safetensors") for name in filenames):
        return None

    files = []
    for sibling in runtime_files:
        filename = sibling["rfilename"]
        files.append({
            "path": filename,
            "role": file_role(filename),
            "sha256": sibling.get("sha256", ""),
            "sizeBytes": sibling.get("size", 0) or 0,
            "source": {
                "type": "hf_snapshot",
                "repoId": repo_id,
                "revision": "main",
                "filename": filename,
            },
            "downloadUrl": client.file_url(repo_id, filename),
        })

    card_data = model.get("cardData") or {}
    quantization = repo_name.rsplit("-", 1)[-1].lower()
    return {
        "modelId": repo_name.lower(),
        "moduleId": spec.module_id,
        "version": "1.0.0",
        "platforms": ["ios", "macos"],
        "files": files,
        "recommendedEngines": ["mlx"],
        "backendKind": spec.backend_kind,
        "capabilities": {
            "supportedModules": list(spec.supported_modules),
            "taskProfiles": ["chat", "tool-calling", "agent"],
            "promptProfiles": {
                "chat": f"{spec.prompt_prefix}.chat.v1",
                "tool-calling": f"{spec.prompt_prefix}.tools.v1",
                "agent": f"{spec.prompt_prefix}.agent.v1",
            },
            "supportsToolCalling": True,
        },
        "constraints": {
            "minOs": "17.0.0",
            "minRamMB": spec.min_ram_mb,
            "supportsStreaming": True,
        },
        "license": card_data.get("license", "unknown"),
        "_meta": {
            "family": spec.name,
            "quant_type": quantization,
            "hfRepoId": repo_id,
            "hfLastModified": model.get("lastModified", ""),
            "hfDownloads": model.get("downloads", 0),
            "totalBytes": sum(file["sizeBytes"] for file in files),
        },
    }


def fetch_family(spec: FamilySpec, client: HubClient) -> list[dict[str, Any]]:
    candidates = client.search_models(
        search=spec.search,
        author="mlx-community",
        library="mlx",
        model_filter="safetensors",
        limit=200,
    )
    manifests = []
    for candidate in candidates:
        repo_id = candidate.get("id", "")
        repo_name = repo_id.rsplit("/", 1)[-1]
        if not spec.repo_pattern.fullmatch(repo_name):
            continue
        manifest = build_manifest(client.model_info(repo_id), spec, client)
        if manifest:
            manifests.append(manifest)
    return sorted(manifests, key=lambda item: item["modelId"])


def write_catalog(path: Path, manifests: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_preserved_chat_models(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    existing = json.loads(path.read_text(encoding="utf-8"))
    return [
        manifest
        for manifest in existing
        if manifest.get("modelId", "").startswith("qwen3-")
    ]


def fetch_catalogs(
    chat_output: Path,
    shared_output: Path,
    client: HubClient | None = None,
) -> None:
    hub = client or HuggingFaceHubClient()
    by_output: dict[str, list[dict[str, Any]]] = {
        "chat": load_preserved_chat_models(chat_output),
        "shared": [],
    }
    for spec in FAMILIES:
        manifests = fetch_family(spec, hub)
        by_output[spec.output].extend(manifests)
        print(f"{spec.name}: {len(manifests)} models")

    write_catalog(chat_output, sorted(by_output["chat"], key=lambda item: item["modelId"]))
    write_catalog(shared_output, sorted(by_output["shared"], key=lambda item: item["modelId"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch MLX Gemma 4 and Qwen 3.5 Chat/Agent model manifests from Hugging Face."
    )
    parser.add_argument("--chat-output", type=Path, default=DEFAULT_CHAT_OUTPUT)
    parser.add_argument("--shared-output", type=Path, default=DEFAULT_SHARED_OUTPUT)
    args = parser.parse_args()
    fetch_catalogs(args.chat_output, args.shared_output)


if __name__ == "__main__":
    main()
