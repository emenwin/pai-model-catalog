#!/usr/bin/env python3
from __future__ import annotations

"""
fetch_mlx_model_catalog.py
从 HuggingFace 搜索 API 获取 MLX Whisper 模型列表，
解析每个模型仓库的文件清单，并生成符合 PAI 模型资产治理规范的
分类模型列表文件（Category Model List），供 ModelRegistry 使用。

数据来源：https://huggingface.co/api/models
输出格式参见：docs/1_design/architecture/PAI 模型资产治理.md §3.1.2

用法：
  python3 fetch_mlx_model_catalog.py
  python3 fetch_mlx_model_catalog.py --output /path/to/asr-mlx-models.json
  python3 fetch_mlx_model_catalog.py --raw-output /path/to/raw.json
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# ─────────────────────────────────────────────
# 数据源配置
# ─────────────────────────────────────────────

# HuggingFace Model Search API
HF_MODEL_API = "https://huggingface.co/api/models"

# 搜索条件（满足 mlx-swift 仅支持 safetensors 的约束）：
#   - search=whisper mlx-community：关键词限定 Whisper + mlx-community
#   - library=mlx：限定 MLX 模型
#   - filter=safetensors：限定仓库含 safetensors
#   - limit=1000：一次拉取足够多结果（当前远小于该数量）
HF_SEARCH_PARAMS = {
    "search": "whisper mlx-community",
    "library": "mlx",
    "filter": "safetensors",
    "limit": "1000",
}

# 当前运行时仅支持非 asr 配置格式，抓取阶段默认过滤 `-asr-` 仓库。
EXCLUDE_ASR_MODELS = True

# 文件下载基础 URL，拼接格式：{HF_RESOLVE_BASE}/{repo_id}/resolve/main/{filename}
HF_RESOLVE_BASE  = "https://huggingface.co"

# 默认输出文件（与本脚本同目录）
DEFAULT_OUTPUT     = Path(__file__).parent / "asr-mlx-models.json"
DEFAULT_RAW_OUTPUT = Path(__file__).parent / "mlx_model_catalog_raw.json"

# ─────────────────────────────────────────────
# MLX Bundle 文件规范
# 参见：docs/wiki/mlx-model-catalog.md
# ─────────────────────────────────────────────

# 实际文件结构（通过 HF API 抽样确认）：
#   config.json       - 必需，模型配置（含 tokenizer 词表，内嵌于 config）
#   weights.safetensors / model.safetensors - 必需（mlx-swift 支持）
#
# 注意：不再接受 weights.npz（mlx-swift 不支持）。

# 权重文件候选名（至少存在一个即满足要求）
WEIGHT_FILES = {"weights.safetensors", "model.safetensors"}

# 必需文件：config.json 必须存在，权重文件至少存在一个
# 注：has_required_files() 中单独处理权重文件的"至少一个"逻辑
REQUIRED_CONFIG = {"config.json"}

# 可选文件：存在时一并记录，供 Registry 下载完整 Bundle
OPTIONAL_FILES = {
    "tokenizer.json",           # 新版格式（部分模型有）
    "merges.txt",               # BPE tokenizer 合并规则
    "vocab.json",               # 词表
    "special_tokens_map.json",  # 特殊 token 映射
    "tokenizer_config.json",    # tokenizer 配置
    "added_tokens.json",        # 新增 token 列表
    "preprocessor_config.json", # 音频预处理配置
}

# 文件角色映射 → manifest files[].role 字段
# role 用于 Registry 在 FilePlan 中区分文件用途
FILE_ROLE_MAP = {
    "weights.safetensors":     "weights",
    "model.safetensors":       "weights",
    "config.json":             "config",
    "tokenizer.json":          "tokenizer",
    "merges.txt":              "tokenizer",
    "vocab.json":              "tokenizer",
    "special_tokens_map.json": "tokenizer",
    "tokenizer_config.json":   "tokenizer",
    "added_tokens.json":       "tokenizer",
    "preprocessor_config.json":"config",
}

# ─────────────────────────────────────────────
# 模型约束推断
# ─────────────────────────────────────────────

# 模型系列 → 最小内存需求（MB）
# 基于实际模型参数量估算，量化版本在 infer_min_ram_mb() 中减半
FAMILY_RAM_MB = {
    "tiny":   2048,
    "base":   2048,
    "small":  4096,
    "medium": 8192,
    "large":  16384,
    "turbo":  8192,   # large-v3-turbo 系列（蒸馏版，参数量约等于 medium）
}

# 已知系列名（顺序从长到短，避免 'large' 被 'large-v3' 前缀误匹配）
KNOWN_FAMILIES = ["large", "medium", "small", "base", "tiny", "turbo"]

# ─────────────────────────────────────────────
# 网络工具
# ─────────────────────────────────────────────

def fetch_json(url: str) -> dict | list:
    """
    从指定 URL 获取 JSON 数据。
    添加 User-Agent 避免被 HuggingFace 拒绝（HF 对无 UA 的请求可能返回 403）。
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; mlx-catalog-fetcher/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP 错误 {e.code}：{url}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"网络错误：{e.reason}", file=sys.stderr)
        sys.exit(1)


def fetch_search_items() -> list[str]:
    """
    从 HuggingFace Search API 获取模型 ID 列表。
    搜索条件：library=mlx、filter=safetensors、关键词 whisper+mlx-community。
    """
    search_url = f"{HF_MODEL_API}?{urllib.parse.urlencode(HF_SEARCH_PARAMS)}"
    print(f"拉取 Search API：{search_url}")
    data = fetch_json(search_url)
    if not isinstance(data, list):
        print(f"Search API 返回结构异常：{type(data)}", file=sys.stderr)
        sys.exit(1)

    model_ids = []
    seen = set()
    skipped_asr = 0
    for entry in data:
        model_id = entry.get("id", "")
        if not model_id:
            continue
        # 二次约束：确保来自 mlx-community，且确实是 whisper 仓库
        if not model_id.startswith("mlx-community/"):
            continue
        if "whisper" not in model_id.lower():
            continue
        if EXCLUDE_ASR_MODELS and is_asr_repo(extract_repo_name(model_id)):
            skipped_asr += 1
            continue
        if model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)
    if EXCLUDE_ASR_MODELS:
        print(f"Search 预过滤：已排除 asr 仓库 {skipped_asr} 个（规则：名称含 '-asr-'）")
    return model_ids


def fetch_model_detail(model_id: str) -> dict:
    """
    获取单个模型的完整信息（含 siblings 文件列表与 cardData）。

    Collection items 只包含基础字段（id/downloads/likes/lastModified），
    不含 siblings（文件清单），需要单独请求模型详情 API。

    API 端点：GET /api/models/{model_id}
    返回完整 model 对象，包含 siblings 数组。
    """
    url = f"{HF_MODEL_API}/{model_id}"
    print(f"    请求模型详情：{url}")
    return fetch_json(url)

# ─────────────────────────────────────────────
# 模型名称解析
# ─────────────────────────────────────────────

def extract_repo_name(model_id: str) -> str:
    """
    从 HuggingFace model_id 中提取仓库名称（去掉组织前缀）。
    例：'mlx-community/whisper-tiny-mlx' → 'whisper-tiny-mlx'
    """
    return model_id.split("/")[-1]


def is_asr_repo(repo_name: str) -> bool:
    """
    判断仓库是否为 `asr` 版本命名（如 whisper-tiny-asr-fp16）。
    """
    name_lower = repo_name.lower()
    return "-asr-" in name_lower or name_lower.endswith("-asr")


def infer_family(repo_name: str) -> str:
    """
    从仓库名称推断模型系列（tiny/base/small/medium/large/turbo）。
    例：'whisper-large-v3-mlx' → 'large'
         'whisper-tiny-mlx'    → 'tiny'
         'whisper-turbo'       → 'turbo'
    """
    name_lower = repo_name.lower()
    for fam in KNOWN_FAMILIES:
        if fam in name_lower:
            return fam
    # 兜底：取 'whisper-' 后的第一段
    parts = re.split(r"[-.]", repo_name)
    return parts[1] if len(parts) > 1 else parts[0]


def infer_metadata(repo_name: str) -> dict:
    """
    从仓库名称推断语言、量化类型等元数据。

    命名规律（来自 Collection 分析）：
      .en / -en  → english-only（仅英语，不支持多语言转录）
      -fp32      → 全精度 32 位浮点（无量化）
      -8bit      → 8 位量化
      -4bit / -q4 → 4 位量化
      -2bit      → 2 位量化（仅 base 模型）
      无精度后缀  → 默认 16 位浮点（mlx 默认格式）
      -v1/-v2/-v3/-v3-turbo → 版本/变体标识
    """
    name_lower = repo_name.lower()

    # 语言判断：含 .en 或以 -en 结尾则为英语专用
    is_english = (
        ".en" in name_lower
        or "-en-" in name_lower
        or name_lower.endswith("-en")
        or name_lower.endswith("-en-mlx")
    )
    lang = "english" if is_english else "multilingual"

    # 量化类型：匹配 -fp32、-8bit、-4bit、-2bit、-q4 等
    quant_m = re.search(r"-(fp32|q\d+(?:_\d+)?|\d+bit)", name_lower)
    quant_type = quant_m.group(1) if quant_m else None

    # 版本/变体：匹配 -v1、-v2、-v3、-v3-turbo 等
    variant_m = re.search(r"-(v\d+(?:-\w+)?)", name_lower)
    variant = variant_m.group(1) if variant_m else None

    return {
        "lang":       lang,
        "quantized":  quant_type is not None and quant_type != "fp32",
        "quant_type": quant_type,
        "variant":    variant,
    }


def infer_min_ram_mb(family: str, quant_type: str | None) -> int:
    """
    根据模型系列与量化类型推断最小内存需求（MB）。
    量化模型内存占用约为原始模型的 50%（保守估算，实际可能更低）。
    fp32 全精度模型内存占用约为默认（fp16）的 2 倍。
    """
    base_ram = FAMILY_RAM_MB.get(family, 4096)
    if quant_type == "fp32":
        return base_ram * 2
    if quant_type and quant_type != "fp32":
        return max(base_ram // 2, 1024)
    return base_ram

# ─────────────────────────────────────────────
# 文件清单解析
# ─────────────────────────────────────────────

def parse_siblings(siblings: list[dict], repo_id: str) -> list[dict]:
    """
    从 HuggingFace siblings 列表中提取 MLX Bundle 相关文件，
    构建 manifest files[] 数组。

    HF siblings 每项格式：
      {"rfilename": "weights.npz", "size": 74400000, ...}

    实际文件结构（抽样确认）：
      - config.json        必需，所有模型均有
      - weights.npz        必需，大多数模型（旧版格式）
      - weights.safetensors 必需，新版模型（如 large-v3-turbo）
      两种权重格式互斥，至少存在一个

    返回的每个文件条目格式（manifest files[] 规范 + downloadUrl 扩展）：
      {
        "path":        "weights.npz",
        "role":        "weights",
        "sha256":      "",          # HF API 不直接提供 sha256，留空；Registry 下载后校验填充
        "sizeBytes":   74400000,
        "downloadUrl": "https://huggingface.co/mlx-community/xxx/resolve/main/weights.npz"
      }

    注：downloadUrl 是对 manifest 规范的扩展字段，供 Task-4b03（下载源 Schema 扩展）使用。
    """
    target_files = WEIGHT_FILES | REQUIRED_CONFIG | OPTIONAL_FILES
    result = []

    for s in siblings:
        fname = s.get("rfilename", "")
        if fname not in target_files:
            continue

        result.append({
            "path":        fname,
            "role":        FILE_ROLE_MAP.get(fname, "other"),
            "sha256":      "",
            "sizeBytes":   s.get("size", 0),
            "downloadUrl": f"{HF_RESOLVE_BASE}/{repo_id}/resolve/main/{fname}",
        })

    return result


def has_required_files(file_entries: list[dict]) -> bool:
    """
    检查文件列表是否满足 MLX Bundle 最低要求：
      1. config.json 必须存在
      2. 权重文件至少存在一个（weights.npz 或 weights.safetensors）

    对应 MLXWhisperBundle.validateBundle() 的校验逻辑。
    """
    present = {f["path"] for f in file_entries}
    has_config  = REQUIRED_CONFIG.issubset(present)
    has_weights = bool(present & WEIGHT_FILES)
    return has_config and has_weights

# ─────────────────────────────────────────────
# Manifest 构建
# ─────────────────────────────────────────────

def build_manifest(model: dict) -> dict | None:
    """
    将 HuggingFace 模型详情转换为 PAI Manifest 格式。

    Manifest 规范参见：docs/1_design/architecture/PAI 模型资产治理.md §2.1

    返回 None 表示该模型不满足 MLX Bundle 要求（缺少必需文件），调用方应跳过。
    """
    repo_id   = model.get("id", "")           # 例：'mlx-community/whisper-tiny-mlx'
    repo_name = extract_repo_name(repo_id)    # 例：'whisper-tiny-mlx'
    siblings  = model.get("siblings", [])

    # 解析文件清单
    file_entries = parse_siblings(siblings, repo_id)

    # 过滤：缺少必需文件的模型不纳入 catalog
    if not has_required_files(file_entries):
        present = {f["path"] for f in file_entries}
        print(f"  ⚠ 跳过 {repo_name}：缺少 config.json 或权重文件（有 {present}）")
        return None

    # 过滤：当前运行时暂不支持 `asr` 风格仓库
    if EXCLUDE_ASR_MODELS and is_asr_repo(repo_name):
        print(f"  ⚠ 跳过 {repo_name}：asr 版本模型（当前运行时仅支持非 asr）")
        return None

    # 推断元数据
    family      = infer_family(repo_name)
    meta        = infer_metadata(repo_name)

    # 过滤英语专属模型（如 *.en / *-en-*）
    if meta["lang"] == "english":
        print(f"  ⚠ 跳过 {repo_name}：英语专属模型")
        return None

    min_ram     = infer_min_ram_mb(family, meta["quant_type"])
    total_bytes = sum(f["sizeBytes"] for f in file_entries)

    # 从 HF cardData 获取 license（若模型卡片有声明）
    card_data   = model.get("cardData", {}) or {}
    license_str = card_data.get("license", "unknown")

    return {
        # ── 资产标识（Registry 主键）──
        "modelId":  repo_name,       # 例：'whisper-tiny-mlx'
        "moduleId": "module.asr",    # 固定为 ASR 模块
        "version":  "1.0.0",         # HF 无版本概念，默认 1.0.0

        # ── 平台支持 ──
        # MLX 依赖 Apple Silicon，仅支持 iOS 17+ / macOS 14+
        "platforms": ["ios", "macos"],

        # ── 文件清单 ──
        # 包含 downloadUrl 扩展字段，供 Task-4b03 下载源 Schema 使用
        "files": file_entries,

        # ── 引擎推荐 ──
        # Registry resolve 阶段会将 mlx 映射到 .mlxSwift backend
        "recommendedEngines": ["mlx"],

        # ── 运行约束 ──
        "constraints": {
            "minOs":            "17.0.0",  # MLX Swift 最低系统要求
            "minRamMB":         min_ram,
            "supportsStreaming": False,    # MLX Whisper 当前不支持流式输出
        },

        "license": license_str,

        # ── 附加元数据（非 manifest 规范字段，供 UI/Policy 层使用）──
        "_meta": {
            "family":         family,
            "lang":           meta["lang"],
            "quantized":      meta["quantized"],
            "quant_type":     meta["quant_type"],
            "variant":        meta["variant"],
            "totalBytes":     total_bytes,
            "hfRepoId":       repo_id,
            "hfLastModified": model.get("lastModified", ""),
            "hfDownloads":    model.get("downloads", 0),
        },
    }

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def fetch_mlx_catalog(output_path: Path, raw_output_path: Path | None = None) -> None:
    """
    完整抓取流程：
    1. 从 Search API 获取模型 ID 列表（library=mlx, filter=safetensors）
    2. 逐个请求模型详情 API，获取 siblings 文件清单
    3. 过滤：只保留包含完整 safetensors Bundle、非英语专属、且非 asr 版本的模型
    4. 构建符合 PAI Manifest 规范的条目列表
    5. 写入分类模型列表文件（Category Model List）
    """
    print("运行时过滤说明：")
    print("  - 仅保留 safetensors 模型")
    print("  - 过滤英语专属模型（*.en）")
    if EXCLUDE_ASR_MODELS:
        print("  - 过滤 asr 版本模型（名称含 '-asr-'）")
    print("")

    # ── Step 1：获取 Search API 模型 ID 列表 ──
    model_ids = fetch_search_items()
    print(f"Search API 命中 {len(model_ids)} 个候选模型\n")

    # ── Step 2：逐个拉取模型详情（含文件清单）──
    print("拉取模型详情...")
    raw_models = []
    for i, model_id in enumerate(model_ids, 1):
        repo_name = extract_repo_name(model_id)
        print(f"  [{i}/{len(model_ids)}] {repo_name}")
        detail = fetch_model_detail(model_id)
        raw_models.append(detail)

    # 可选：保存原始 API 响应（用于调试与离线分析）
    if raw_output_path:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(
            json.dumps(raw_models, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n原始数据已保存至：{raw_output_path}")

    # ── Step 3 & 4：过滤 + 构建 manifest ──
    print("\n解析文件清单，构建 manifest...")
    manifests = []
    skipped   = 0
    for model in raw_models:
        manifest = build_manifest(model)
        if manifest is None:
            skipped += 1
            continue
        manifests.append(manifest)

    # 按 modelId 字典序排序（与 Registry resolve 的 deterministic tie-break 规则一致）
    manifests.sort(key=lambda m: m["modelId"])

    # ── Step 5：写入分类模型列表文件 ──
    # 格式：JSON 数组，每项为一个完整 manifest（内联，无需再逐个拉取 <modelId>.json）
    # 对应架构设计 §3.1.2 Category Model List 规范
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n完成！")
    print(f"  有效模型：{len(manifests)} 个")
    print(f"  跳过（缺少 safetensors 必需文件 / 英语专属 / asr 版本）：{skipped} 个")
    print(f"  已保存至：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 HuggingFace Search API 获取 MLX Whisper 模型目录（仅 safetensors，过滤 en 专属与 asr 版本）",
        epilog=(
            "输出文件为 PAI Category Model List 格式（asr-mlx-models.json），\n"
            "供 ModelRegistry.refreshCatalog() 批量 register 使用。\n"
            "数据来源：https://huggingface.co/api/models"
        ),
    )
    parser.add_argument(
        "--output", metavar="FILE", type=Path, default=DEFAULT_OUTPUT,
        help=f"输出 JSON 文件路径（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--raw-output", metavar="FILE", type=Path, default=None,
        help="保存原始 HuggingFace API 响应（用于调试，默认不保存）",
    )
    args = parser.parse_args()
    fetch_mlx_catalog(args.output, args.raw_output)


if __name__ == "__main__":
    main()
