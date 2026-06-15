#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


class HuggingFaceHubClient:
    """Small adapter around huggingface_hub used by catalog generators."""

    def __init__(self, token: str | None = None) -> None:
        try:
            from huggingface_hub import HfApi, hf_hub_url
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency: install with "
                "`python3 -m pip install -r Scripts/requirements.txt`."
            ) from error

        self._api = HfApi(token=token)
        self._hf_hub_url = hf_hub_url

    def search_models(
        self,
        *,
        search: str,
        author: str | None = None,
        library: str | None = None,
        model_filter: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        filters = [value for value in (library, model_filter) if value]
        models = self._api.list_models(
            search=search,
            author=author,
            filter=filters or None,
            limit=limit,
            full=True,
            cardData=True,
        )
        return [self._model_to_dict(model) for model in models]

    def model_info(self, repo_id: str) -> dict[str, Any]:
        model = self._api.model_info(repo_id, files_metadata=True)
        return self._model_to_dict(model)

    def file_url(self, repo_id: str, filename: str, revision: str = "main") -> str:
        return self._hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)

    @classmethod
    def _model_to_dict(cls, model: Any) -> dict[str, Any]:
        card_data = getattr(model, "card_data", None)
        if card_data is None:
            card_data = getattr(model, "cardData", None)

        return {
            "id": getattr(model, "id", None) or getattr(model, "modelId", ""),
            "lastModified": cls._stringify(getattr(model, "last_modified", None)),
            "downloads": getattr(model, "downloads", 0) or 0,
            "likes": getattr(model, "likes", 0) or 0,
            "library_name": getattr(model, "library_name", None),
            "tags": list(getattr(model, "tags", None) or []),
            "cardData": cls._mapping(card_data),
            "siblings": [
                cls._sibling_to_dict(sibling)
                for sibling in (getattr(model, "siblings", None) or [])
            ],
        }

    @classmethod
    def _sibling_to_dict(cls, sibling: Any) -> dict[str, Any]:
        lfs = cls._mapping(getattr(sibling, "lfs", None))
        sha256 = lfs.get("sha256", "")
        return {
            "rfilename": getattr(sibling, "rfilename", ""),
            "size": getattr(sibling, "size", None) or lfs.get("size", 0) or 0,
            "sha256": sha256,
        }

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    @staticmethod
    def _stringify(value: Any) -> str:
        return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def repository_ids(models: Iterable[dict[str, Any]]) -> list[str]:
    return [model["id"] for model in models if model.get("id")]
