#!/usr/bin/env python3
"""
ggml_model_manager.py
Whisper GGML 模型管理工具

功能：
  - 列出所有可用模型及详细信息（从 model_catalog.json 读取）
  - 下载指定模型（支持 --coreml 同时下载 CoreML 编码器）
  - 检查本地已下载状态

前置条件：
  先运行 fetch_model_catalog.py 生成 model_catalog.json

用法：
  python3 ggml_model_manager.py list
  python3 ggml_model_manager.py list --detail
  python3 ggml_model_manager.py download tiny
  python3 ggml_model_manager.py download tiny --coreml
  python3 ggml_model_manager.py download tiny --output /path/to/dir
"""

import argparse
import json
import sys
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

# model_catalog.json 默认路径（与本脚本同目录，由 fetch_model_catalog.py 生成）
CATALOG_FILE = Path(__file__).parent / "model_catalog.json"

# ─────────────────────────────────────────────
# Catalog 加载
# ─────────────────────────────────────────────

def load_catalog(catalog_path: Path = CATALOG_FILE) -> list[dict]:
    """
    从 JSON 文件加载模型目录。
    若文件不存在，提示用户先运行 fetch_model_catalog.py。
    """
    if not catalog_path.exists():
        print(f"错误：找不到模型目录文件 {catalog_path}")
        print("请先运行：python3 fetch_model_catalog.py")
        sys.exit(1)
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    return data.get("models", [])


def get_model_by_name(name: str, catalog: list[dict]) -> Optional[dict]:
    """根据模型名称在目录中查找，未找到返回 None"""
    for m in catalog:
        if m["name"] == name:
            return m
    return None

# ─────────────────────────────────────────────
# 路径与状态工具
# ─────────────────────────────────────────────

def get_bin_local_path(model_name: str, output_dir: Path) -> Path:
    """返回 GGML 权重文件的本地路径"""
    return output_dir / f"ggml-{model_name}.bin"


def get_coreml_local_path(model_name: str, output_dir: Path) -> Path:
    """返回 CoreML 编码器目录的本地路径（解压后为目录）"""
    return output_dir / f"ggml-{model_name}-encoder.mlmodelc"


def is_bin_downloaded(model_name: str, output_dir: Path) -> bool:
    """检查 GGML 权重文件是否已下载"""
    return get_bin_local_path(model_name, output_dir).exists()


def is_coreml_downloaded(model_name: str, output_dir: Path) -> bool:
    """检查 CoreML 编码器目录是否已解压到本地"""
    return get_coreml_local_path(model_name, output_dir).exists()


def format_size(size_mb: int) -> str:
    """将 MB 数值格式化为可读字符串"""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb} MB"


# ─────────────────────────────────────────────
# 功能模块：列出模型
# ─────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> None:
    """
    列出所有可用模型。
    --detail 模式：显示完整信息表格（含 CoreML 支持列）
    默认模式：按系列分组简洁展示
    """
    catalog    = load_catalog()
    output_dir = Path(args.output) if args.output else Path(__file__).parent

    if args.detail:
        _list_detail(catalog, output_dir)
    else:
        _list_simple(catalog, output_dir)


def _list_simple(catalog: list[dict], output_dir: Path) -> None:
    """简洁模式：按系列分组，已下载标 ✓"""
    print("\n可用模型（按系列分组）：\n")
    current_family = None
    for m in catalog:
        if m["family"] != current_family:
            if current_family is not None:
                print()
            print(f"  [{m['family']}]", end="")
            current_family = m["family"]
        mark = " ✓" if is_bin_downloaded(m["name"], output_dir) else ""
        print(f"  {m['name']}{mark}", end="")
    print("\n")
    print("提示：使用 --detail 查看详细信息（含 CoreML 支持），✓ 表示已下载\n")


def _list_detail(catalog: list[dict], output_dir: Path) -> None:
    """详细模式：表格展示，含 CoreML 列"""
    col_name   = 26
    col_lang   = 8
    col_quant  = 8
    col_bin    = 6   # 本地 bin
    col_coreml = 9   # CoreML 支持 + 本地状态

    header = (
        f"{'模型名称':<{col_name}}"
        f"{'语言':<{col_lang}}"
        f"{'量化':<{col_quant}}"
        f"{'本地':^{col_bin}}"
        f"{'CoreML':^{col_coreml}}"
        f"备注"
    )
    print("\n" + header)
    print("─" * 88)

    current_family = None
    for m in catalog:
        if m["family"] != current_family:
            if current_family is not None:
                print()
            current_family = m["family"]

        bin_mark    = "✓" if is_bin_downloaded(m["name"], output_dir) else "─"
        lang_str    = "英文" if m["lang"] == "english" else "多语言"
        quant_str   = m["quant_type"] if m["quant_type"] else "─"

        # CoreML 列：支持且已下载 → ✓，支持未下载 → 可用，不支持 → ─
        if m["coreml"]:
            coreml_str = "✓" if is_coreml_downloaded(m["name"], output_dir) else "可用"
        else:
            coreml_str = "─"

        notes = []
        if m["tdrz"]:
            notes.append("tinydiarize")
        note_str = ", ".join(notes)

        print(
            f"{m['name']:<{col_name}}"
            f"{lang_str:<{col_lang}}"
            f"{quant_str:<{col_quant}}"
            f"{bin_mark:^{col_bin}}"
            f"{coreml_str:^{col_coreml}}"
            f"{note_str}"
        )
    print()


# ─────────────────────────────────────────────
# 功能模块：下载模型
# ─────────────────────────────────────────────

def cmd_download(args: argparse.Namespace) -> None:
    """
    下载指定模型。
    --coreml：同时下载并解压 CoreML 编码器（用于 Apple Neural Engine 加速）
    """
    catalog = load_catalog()
    model   = get_model_by_name(args.model, catalog)
    if model is None:
        print(f"错误：未知模型 '{args.model}'")
        print("使用 'list' 命令查看所有可用模型。")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 下载 GGML 权重文件 ──
    bin_path = get_bin_local_path(args.model, output_dir)
    if bin_path.exists():
        print(f"模型 '{args.model}' 已存在：{bin_path}，跳过。")
    else:
        print(f"正在下载 GGML 模型 '{args.model}'...")
        print(f"  来源：{model['bin_url']}")
        print(f"  目标：{bin_path}")
        _download_with_progress(model["bin_url"], bin_path)
        print(f"\n下载完成：{bin_path}")

    # ── 下载 CoreML 编码器（可选）──
    if args.coreml:
        if not model["coreml"]:
            print(f"\n注意：模型 '{args.model}' 没有对应的 CoreML 编码器（量化版本不支持）。")
        else:
            coreml_dir = get_coreml_local_path(args.model, output_dir)
            if coreml_dir.exists():
                print(f"\nCoreML 编码器已存在：{coreml_dir}，跳过。")
            else:
                print(f"\n正在下载 CoreML 编码器...")
                print(f"  来源：{model['coreml_url']}")
                zip_path = output_dir / f"ggml-{args.model}-encoder.mlmodelc.zip"
                _download_with_progress(model["coreml_url"], zip_path)
                print(f"\n正在解压 CoreML 编码器...")
                _unzip_coreml(zip_path, output_dir)
                print(f"解压完成：{coreml_dir}")

    # ── 使用提示 ──
    print(f"\n使用示例：")
    if args.coreml and model["coreml"]:
        print(f"  whisper-cli -m {bin_path} --coreml-encoder-path {get_coreml_local_path(args.model, output_dir)} -f samples/jfk.wav")
    else:
        print(f"  whisper-cli -m {bin_path} -f samples/jfk.wav")
    print()


def _download_with_progress(url: str, dest: Path) -> None:
    """
    执行实际下载，终端显示进度条。
    使用 urllib 标准库，无需额外依赖。
    下载失败时自动清理不完整文件。
    """
    def _hook(block_count: int, block_size: int, total_size: int) -> None:
        """urlretrieve 进度回调"""
        if total_size <= 0:
            downloaded = block_count * block_size
            print(f"\r  已下载：{downloaded / 1024 / 1024:.1f} MB", end="", flush=True)
            return
        downloaded = min(block_count * block_size, total_size)
        percent    = downloaded / total_size * 100
        bar_len    = 40
        filled     = int(bar_len * downloaded / total_size)
        bar        = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {percent:5.1f}%  "
            f"{downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB",
            end="", flush=True,
        )

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_hook)
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"\n下载失败：{e}")
        print("请检查网络连接后重试。")
        sys.exit(1)


def _unzip_coreml(zip_path: Path, output_dir: Path) -> None:
    """
    解压 CoreML 编码器 zip 包到 output_dir，解压完成后删除 zip 文件。
    zip 内包含 .mlmodelc 目录（CoreML 编译模型包）。
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
    except zipfile.BadZipFile as e:
        print(f"\n解压失败（文件损坏）：{e}")
        zip_path.unlink(missing_ok=True)
        sys.exit(1)
    finally:
        # 无论成功与否，清理 zip 文件
        if zip_path.exists():
            zip_path.unlink()


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="ggml_model_manager.py",
        description="Whisper GGML 模型管理工具（需先运行 fetch_model_catalog.py）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python3 fetch_model_catalog.py          # 先抓取模型目录\n"
            "  python3 ggml_model_manager.py list\n"
            "  python3 ggml_model_manager.py list --detail\n"
            "  python3 ggml_model_manager.py download tiny\n"
            "  python3 ggml_model_manager.py download tiny --coreml\n"
            "  python3 ggml_model_manager.py download tiny --output ./models\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── list 子命令 ──
    p_list = subparsers.add_parser("list", help="列出所有可用模型")
    p_list.add_argument(
        "--detail", action="store_true",
        help="显示详细信息（语言、量化类型、CoreML 支持等）",
    )
    p_list.add_argument(
        "--output", metavar="DIR",
        help="本地模型目录（用于检测已下载状态，默认为脚本所在目录）",
    )
    p_list.set_defaults(func=cmd_list)

    # ── download 子命令 ──
    p_dl = subparsers.add_parser("download", help="下载指定模型")
    p_dl.add_argument("model", help="模型名称（如 tiny、base、large-v3-turbo）")
    p_dl.add_argument(
        "--coreml", action="store_true",
        help="同时下载 CoreML 编码器（用于 Apple Neural Engine 加速，仅非量化模型支持）",
    )
    p_dl.add_argument(
        "--output", metavar="DIR",
        help="保存目录（默认为脚本所在目录）",
    )
    p_dl.set_defaults(func=cmd_download)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

