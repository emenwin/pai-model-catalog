#!/usr/bin/env python3
"""
fetch_model_catalog.py
从 HuggingFace 获取 whisper.cpp 模型列表，解析所有 GGML 模型与 CoreML 编码器，
并将结果保存到 model_catalog.json，供 ggml_model_manager.py 使用。

用法：
  python3 fetch_model_catalog.py
  python3 fetch_model_catalog.py --output /path/to/model_catalog.json
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub_client import HuggingFaceHubClient

# ─────────────────────────────────────────────
# 数据源配置
# ─────────────────────────────────────────────

# 主仓库
HF_REPO_ID   = "ggerganov/whisper.cpp"
HF_RESOLVE   = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
HF_PAGE_URL  = "https://huggingface.co/ggerganov/whisper.cpp"

# tinydiarize 独立仓库（说话人分离版本）
TDRZ_REPO_ID = "akashmjn/tinydiarize-whisper.cpp"
TDRZ_RESOLVE = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp/resolve/main"

# 默认输出文件（与本脚本同目录）
DEFAULT_OUTPUT = Path(__file__).parent / "model_catalog.json"

# 已知模型系列（用于 family 推断，顺序从长到短避免前缀误匹配）
KNOWN_FAMILIES = ["large", "medium", "small", "base", "tiny"]

# ─────────────────────────────────────────────
# 文件名解析
# ─────────────────────────────────────────────

def parse_filename(filename: str) -> dict | None:
    """
    从文件名解析类型与模型名称。
    支持：
      ggml-{name}.bin                    → GGML 权重文件
      ggml-{name}-encoder.mlmodelc.zip   → CoreML 编码器（Apple Neural Engine 加速）
    返回 None 表示不是目标文件。
    """
    # GGML 权重文件
    m = re.match(r"^ggml-(.+)\.bin$", filename)
    if m:
        return {"type": "bin", "model_name": m.group(1)}

    # CoreML 编码器（zip 压缩包）
    m = re.match(r"^ggml-(.+)-encoder\.mlmodelc\.zip$", filename)
    if m:
        return {"type": "coreml", "model_name": m.group(1)}

    return None

# ─────────────────────────────────────────────
# 元数据推断（从模型名称）
# ─────────────────────────────────────────────

def infer_family(name: str) -> str:
    """从模型名称推断所属系列（tiny/base/small/medium/large）"""
    for fam in KNOWN_FAMILIES:
        if name.startswith(fam):
            return fam
    # 兜底：取第一个分隔符前的部分
    return re.split(r"[-.]", name)[0]


def infer_metadata(name: str, tdrz: bool = False) -> dict:
    """
    从模型名称推断语言、量化类型等元数据。
    规则：
      .en  → english-only
      -q*  → quantized（量化版本）
      tdrz → tinydiarize（说话人分离）
    """
    lang       = "english" if ".en" in name else "multilingual"
    quant_m    = re.search(r"-(q\d+_\d+)", name)
    quant_type = quant_m.group(1) if quant_m else None
    return {
        "lang":      lang,
        "quantized": quant_type is not None,
        "quant_type": quant_type,
        "tdrz":      tdrz,
    }

# ─────────────────────────────────────────────
# 目录构建
# ─────────────────────────────────────────────

def parse_siblings(siblings: list[dict]) -> tuple[dict, set]:
    """
    从 HuggingFace API 返回的 siblings 列表中提取：
      bin_map:    {model_name: filename}   所有 GGML 权重文件
      coreml_set: {model_name}             有 CoreML 编码器的模型集合
    """
    bin_map    = {}
    coreml_set = set()
    for s in siblings:
        fname  = s.get("rfilename", "")
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        name = parsed["model_name"]
        if parsed["type"] == "bin":
            bin_map[name] = fname
        elif parsed["type"] == "coreml":
            coreml_set.add(name)
    return bin_map, coreml_set


def build_model_entries(
    bin_map: dict,
    coreml_set: set,
    resolve_base: str,
    tdrz: bool = False,
) -> list[dict]:
    """
    将 bin_map 与 coreml_set 组装为标准模型条目列表。
    每个条目包含完整的下载 URL，供 ggml_model_manager.py 直接使用。
    """
    entries = []
    for name in sorted(bin_map):
        has_coreml = name in coreml_set
        meta = infer_metadata(name, tdrz=tdrz)
        entries.append({
            "name":         name,
            "family":       infer_family(name),
            **meta,
            "coreml":       has_coreml,
            # 下载 URL（直接可用，无需客户端拼接）
            "bin_url":      f"{resolve_base}/{bin_map[name]}",
            "coreml_url":   f"{resolve_base}/ggml-{name}-encoder.mlmodelc.zip" if has_coreml else None,
        })
    return entries

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def fetch_catalog(output_path: Path) -> None:
    """
    完整抓取流程：
    1. 拉取主仓库 API → 解析 GGML + CoreML 文件
    2. 拉取 tinydiarize 仓库 API → 解析 tdrz 模型
    3. 合并并写入 JSON 文件
    """
    all_entries: list[dict] = []
    client = HuggingFaceHubClient()

    # ── 主仓库 ──
    print(f"拉取主仓库：{HF_REPO_ID}")
    data     = client.model_info(HF_REPO_ID)
    siblings = data.get("siblings", [])
    bin_map, coreml_set = parse_siblings(siblings)
    entries  = build_model_entries(bin_map, coreml_set, HF_RESOLVE, tdrz=False)
    print(f"  → {len(entries)} 个模型，{len(coreml_set)} 个 CoreML 编码器")
    all_entries.extend(entries)

    # ── tinydiarize 仓库 ──
    print(f"拉取 tinydiarize 仓库：{TDRZ_REPO_ID}")
    tdrz_data     = client.model_info(TDRZ_REPO_ID)
    tdrz_siblings = tdrz_data.get("siblings", [])
    tdrz_bin_map, tdrz_coreml_set = parse_siblings(tdrz_siblings)
    tdrz_entries  = build_model_entries(tdrz_bin_map, tdrz_coreml_set, TDRZ_RESOLVE, tdrz=True)
    print(f"  → {len(tdrz_entries)} 个 tinydiarize 模型")
    all_entries.extend(tdrz_entries)

    # ── 写入文件 ──
    catalog = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source":     HF_PAGE_URL,
        "total":      len(all_entries),
        "models":     all_entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成！共 {len(all_entries)} 个模型，已保存至：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 抓取 whisper.cpp 模型目录",
        epilog="输出文件供 ggml_model_manager.py 使用。",
    )
    parser.add_argument(
        "--output", metavar="FILE", type=Path, default=DEFAULT_OUTPUT,
        help=f"输出 JSON 文件路径（默认：{DEFAULT_OUTPUT}）",
    )
    args = parser.parse_args()
    fetch_catalog(args.output)


if __name__ == "__main__":
    main()
