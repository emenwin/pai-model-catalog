import json
import unittest
from pathlib import Path

from fetch_llm_model_catalog import FAMILIES, build_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class FakeHubClient:
    def file_url(self, repo_id, filename, revision="main"):
        return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


class LLMModelCatalogTests(unittest.TestCase):
    def test_qwen_manifest_includes_tool_calling_and_runtime_files(self):
        model = {
            "id": "mlx-community/Qwen3.5-0.8B-4bit",
            "cardData": {"license": "apache-2.0"},
            "siblings": [
                {"rfilename": "config.json", "size": 10},
                {"rfilename": "chat_template.jinja", "size": 20},
                {"rfilename": "model.safetensors.index.json", "size": 30},
                {"rfilename": "model.safetensors", "size": 40, "sha256": "abc"},
                {"rfilename": "README.md", "size": 50},
            ],
        }

        manifest = build_manifest(model, FAMILIES[0], FakeHubClient())

        self.assertIsNotNone(manifest)
        self.assertTrue(manifest["capabilities"]["supportsToolCalling"])
        self.assertIn("module.agent", manifest["capabilities"]["supportedModules"])
        self.assertEqual(
            [item["path"] for item in manifest["files"]],
            [
                "chat_template.jinja",
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
            ],
        )
        self.assertEqual(manifest["files"][2]["sha256"], "abc")

    def test_rejects_unrelated_gemma_repository(self):
        model = {
            "id": "mlx-community/text-to-cypher-gemma-3-4b",
            "siblings": [
                {"rfilename": "config.json", "size": 10},
                {"rfilename": "model.safetensors", "size": 40},
            ],
        }

        self.assertIsNone(build_manifest(model, FAMILIES[1], FakeHubClient()))

    def test_generated_catalogs_keep_chat_template_and_agent_capability(self):
        manifests = []
        for filename in ("llm-chat-mlx-models.json", "llm-shared-mlx-models.json"):
            manifests.extend(json.loads((REPOSITORY_ROOT / filename).read_text(encoding="utf-8")))

        target_manifests = [
            manifest
            for manifest in manifests
            if "qwen3.5" in manifest["modelId"] or "gemma-4" in manifest["modelId"]
        ]

        self.assertEqual(len(target_manifests), 12)
        for manifest in target_manifests:
            filenames = {item["path"] for item in manifest["files"]}
            self.assertIn("chat_template.jinja", filenames)
            self.assertIn("model.safetensors.index.json", filenames)
            self.assertIn("module.llm.chat", manifest["capabilities"]["supportedModules"])
            self.assertIn("module.agent", manifest["capabilities"]["supportedModules"])
            self.assertTrue(manifest["capabilities"]["supportsToolCalling"])

        self.assertFalse(any("text-to-cypher" in manifest["modelId"] for manifest in manifests))


if __name__ == "__main__":
    unittest.main()
