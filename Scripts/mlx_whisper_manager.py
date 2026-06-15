#!/usr/bin/env python3
"""
mlx_whisper_manager.py
MLX Whisper 模型管理工具

功能：
  - 列出所有可用 MLX Whisper 模型及详细信息（从 asr-mlx-models.json 读取）
  - 下载指定模型（下载 Bundle 内所有文件到目标目录）
  - 检查本地已下载状态

前置条件：
  先运行 fetch_mlx_model_catalog.py 生成 asr-mlx-models.json

用法：
  python3 mlx_whisper_manager.py list
  python3 mlx_whisper_manager.py list --detail
  python3 mlx_whisper_manager.py list --family small
  python3 mlx_whisper_manager.py download whisper-tiny-mlx
  python3 mlx_whisper_manager.py download whisper-tiny-mlx --output /path/to/dir
  python3 mlx_whisper_manager.py info whisper-tiny-mlx
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

# asr-mlx-models.json 默认路径（与本脚本同目录，由 fetch_mlx_model_catalog.py 生成）
CATALOG_FILE = Path(__file__).parent / "asr-mlx-models.json"

# 模型默认下载目录（每个模型存放在独立子目录 <output>/<modelId>/）
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "mlx-models"

# ─────────────────────────────────────────────
# Catalog 加载
# ─────────────────────────────────────────────

def load_catalog(catalog_path: Path = CATALOG_FILE) -> list[dict]:
    """
    从 asr-mlx-models.json 加载模型目录（Category Model List 格式）。
    若文件不存在，提示用户先运行 fetch_mlx_model_catalog.py。
    """
    if not catalog_path.exists():
        print(f"错误：找不到模型目录文件 {catalog_path}")
        print("请先运行：python3 fetch_mlx_model_catalog.py")
        sys.exit(1)
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def get_model(model_id: str, catalog: list[dict]) -> Optional[dict]:
    """
    按 modelId 查找模型，未找到返回 None。
    支持完整 ID（whisper-tiny-mlx）或简短名称（tiny-mlx）。
    """
    for m in catalog:
        if m["modelId"] == model_id:
            return m
    # 模糊匹配：允许省略 'whisper-' 前缀
    for m in catalog:
        if m["modelId"].endswith(f"-{model_id}") or m["modelId"] == f"whisper-{model_id}":
            return m
    return None

# ─────────────────────────────────────────────
# 路径与状态工具
# ─────────────────────────────────────────────

def model_dir(model_id: str, output_dir: Path) -> Path:
    """
    返回模型的本地目录路径。
    每个模型存放在独立子目录：<output_dir>/<modelId>/
    例：mlx-models/whisper-tiny-mlx/
    """
    return output_dir / model_id


def is_downloaded(model: dict, output_dir: Path) -> bool:
    """
    检查模型是否已完整下载（所有 files[] 中的文件均存在）。
    """
    mdir = model_dir(model["modelId"], output_dir)
    return all(
        (mdir / f["path"]).exists()
        for f in model["files"]
    )


def downloaded_files(model: dict, output_dir: Path) -> list[str]:
    """返回已下载的文件名列表（用于部分下载状态展示）"""
    mdir = model_dir(model["modelId"], output_dir)
    return [f["path"] for f in model["files"] if (mdir / f["path"]).exists()]


def format_bytes(n: int) -> str:
    """将字节数格式化为可读字符串"""
    if n <= 0:
        return "未知大小"
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def total_size(model: dict) -> int:
    """返回模型所有文件的总字节数"""
    return model.get("_meta", {}).get("totalBytes", 0) or sum(
        f.get("sizeBytes", 0) for f in model["files"]
    )

# ─────────────────────────────────────────────
# 功能模块：列出模型
# ─────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> None:
    """
    列出所有可用 MLX Whisper 模型。
    --detail：显示完整信息表格（含量化类型、内存需求、下载状态）
    --family：按系列过滤（tiny/base/small/medium/large）
    默认：按系列分组简洁展示
    """
    catalog    = load_catalog()
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR

    # 按 family 过滤
    if args.family:
        catalog = [m for m in catalog if m.get("_meta", {}).get("family") == args.family]
        if not catalog:
            print(f"未找到 family='{args.family}' 的模型")
            sys.exit(1)

    if args.detail:
        _list_detail(catalog, output_dir)
    else:
        _list_simple(catalog, output_dir)


def _list_simple(catalog: list[dict], output_dir: Path) -> None:
    """简洁模式：按系列分组，已完整下载标 ✓，部分下载标 ◑"""
    print("\n可用 MLX Whisper 模型（按系列分组）：\n")
    current_family = None
    for m in catalog:
        family = m.get("_meta", {}).get("family", "?")
        if family != current_family:
            if current_family is not None:
                print()
            print(f"  [{family}]")
            current_family = family

        if is_downloaded(m, output_dir):
            mark = " ✓"
        elif downloaded_files(m, output_dir):
            mark = " ◑"  # 部分下载
        else:
            mark = ""

        lang = m.get("_meta", {}).get("lang", "")
        lang_tag = " [en]" if lang == "english" else ""
        print(f"    {m['modelId']}{lang_tag}{mark}")

    print()
    print("提示：✓ 已完整下载  ◑ 部分下载  使用 --detail 查看详细信息\n")


def _list_detail(catalog: list[dict], output_dir: Path) -> None:
    """
    详细模式：表格展示，含量化类型、内存需求、文件大小、下载状态。
    """
    # 列宽
    W_ID    = 32
    W_LANG  = 8
    W_QUANT = 8
    W_RAM   = 10
    W_SIZE  = 10
    W_STATE = 8

    header = (
        f"{'modelId':<{W_ID}}"
        f"{'语言':<{W_LANG}}"
        f"{'量化':<{W_QUANT}}"
        f"{'最低内存':>{W_RAM}}"
        f"{'大小':>{W_SIZE}}"
        f"{'状态':^{W_STATE}}"
    )
    print("\n" + header)
    print("─" * (W_ID + W_LANG + W_QUANT + W_RAM + W_SIZE + W_STATE + 2))

    current_family = None
    for m in catalog:
        meta   = m.get("_meta", {})
        family = meta.get("family", "?")

        if family != current_family:
            if current_family is not None:
                print()
            current_family = family

        lang_str  = "英文" if meta.get("lang") == "english" else "多语言"
        quant_str = meta.get("quant_type") or "─"
        ram_str   = f"{m['constraints']['minRamMB']} MB"
        size_str  = format_bytes(total_size(m))

        # 下载状态
        dl_files = downloaded_files(m, output_dir)
        total_f  = len(m["files"])
        if len(dl_files) == total_f:
            state = "✓ 完整"
        elif dl_files:
            state = f"◑ {len(dl_files)}/{total_f}"
        else:
            state = "─"

        print(
            f"{m['modelId']:<{W_ID}}"
            f"{lang_str:<{W_LANG}}"
            f"{quant_str:<{W_QUANT}}"
            f"{ram_str:>{W_RAM}}"
            f"{size_str:>{W_SIZE}}"
            f"{state:^{W_STATE}}"
        )
    print()

# ─────────────────────────────────────────────
# 功能模块：模型详情
# ─────────────────────────────────────────────

def cmd_info(args: argparse.Namespace) -> None:
    """
    显示指定模型的完整信息（manifest 字段 + 本地状态）。
    """
    catalog    = load_catalog()
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    model      = get_model(args.model_id, catalog)

    if model is None:
        print(f"错误：未找到模型 '{args.model_id}'")
        print("使用 'list' 命令查看所有可用模型。")
        sys.exit(1)

    meta = model.get("_meta", {})
    mdir = model_dir(model["modelId"], output_dir)

    print(f"\n模型：{model['modelId']}")
    print(f"  moduleId    : {model['moduleId']}")
    print(f"  version     : {model['version']}")
    print(f"  platforms   : {', '.join(model['platforms'])}")
    print(f"  engines     : {', '.join(model['recommendedEngines'])}")
    print(f"  minOs       : {model['constraints']['minOs']}")
    print(f"  minRamMB    : {model['constraints']['minRamMB']} MB")
    print(f"  license     : {model['license']}")
    print(f"  语言        : {'英文' if meta.get('lang') == 'english' else '多语言'}")
    print(f"  量化        : {meta.get('quant_type') or '无'}")
    print(f"  HF 仓库     : https://huggingface.co/{meta.get('hfRepoId', '')}")
    print(f"  HF 下载量   : {meta.get('hfDownloads', 0):,}")
    print(f"\n  文件清单：")
    for f in model["files"]:
        local = mdir / f["path"]
        exists = "✓" if local.exists() else "─"
        size   = format_bytes(f.get("sizeBytes", 0))
        print(f"    [{exists}] {f['path']:<30} {size:<12} role={f['role']}")
    print(f"\n  本地目录：{mdir}")
    print(f"  下载状态：{'✓ 完整' if is_downloaded(model, output_dir) else '未完整下载'}\n")

# ─────────────────────────────────────────────
# 功能模块：下载模型
# ─────────────────────────────────────────────

def cmd_download(args: argparse.Namespace) -> None:
    """
    下载指定模型的所有 Bundle 文件到 <output_dir>/<modelId>/ 目录。

    下载策略：
    - 已存在的文件默认跳过（--force 强制重新下载）
    - 每个文件下载完成后显示进度
    - 下载失败时清理不完整文件并退出
    """
    catalog    = load_catalog()
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    model      = get_model(args.model_id, catalog)

    if model is None:
        print(f"错误：未找到模型 '{args.model_id}'")
        print("使用 'list' 命令查看所有可用模型。")
        sys.exit(1)

    mdir = model_dir(model["modelId"], output_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    meta = model.get("_meta", {})
    print(f"\n模型：{model['modelId']}")
    print(f"  语言：{'英文' if meta.get('lang') == 'english' else '多语言'}")
    print(f"  量化：{meta.get('quant_type') or '无（fp16）'}")
    print(f"  最低内存：{model['constraints']['minRamMB']} MB")
    print(f"  目标目录：{mdir}\n")

    files = model["files"]
    downloaded = 0
    skipped    = 0

    for i, f in enumerate(files, 1):
        dest = mdir / f["path"]
        url  = f.get("downloadUrl", "")

        if not url:
            print(f"  [{i}/{len(files)}] ⚠ 跳过 {f['path']}：无下载 URL")
            continue

        if dest.exists() and not args.force:
            print(f"  [{i}/{len(files)}] ─ {f['path']} 已存在，跳过（--force 强制重新下载）")
            skipped += 1
            continue

        print(f"  [{i}/{len(files)}] 下载 {f['path']} ({format_bytes(f.get('sizeBytes', 0))})")
        print(f"        来源：{url}")
        _download_with_progress(url, dest)
        print()
        downloaded += 1

    # ── 下载结果汇总 ──
    print(f"完成！下载 {downloaded} 个，跳过 {skipped} 个")
    if is_downloaded(model, output_dir):
        print(f"✓ 模型已完整下载：{mdir}")
        _print_usage_hint(model, mdir)
    else:
        missing = [f["path"] for f in files if not (mdir / f["path"]).exists()]
        print(f"⚠ 以下文件缺失：{missing}")
    print()


def _print_usage_hint(model: dict, mdir: Path) -> None:
    """打印模型加载提示（供开发调试参考）"""
    weights_file = next(
        (f["path"] for f in model["files"] if f["role"] == "weights"), None
    )
    print(f"\n加载示例（Swift）：")
    print(f'  let bundle = try MLXWhisperBundle.load(')
    print(f'      modelURL: URL(fileURLWithPath: "{mdir}"),')
    print(f'      overrideRoot: nil')
    print(f'  )')
    if weights_file:
        print(f'  // 权重文件：{weights_file}')


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
            print(f"\r        已下载：{downloaded / 1024 / 1024:.1f} MB", end="", flush=True)
            return
        downloaded = min(block_count * block_size, total_size)
        percent    = downloaded / total_size * 100
        bar_len    = 36
        filled     = int(bar_len * downloaded / total_size)
        bar        = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r        [{bar}] {percent:5.1f}%  "
            f"{downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB",
            end="", flush=True,
        )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; mlx-whisper-manager/1.0)"},
    )
    try:
        # urlretrieve 不支持自定义 headers，改用 urlopen + 手动写文件
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            block = 65536  # 64 KB
            downloaded = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    _hook(downloaded // block, block, total)
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"\n        下载失败：{e}")
        print("        请检查网络连接后重试。")
        sys.exit(1)

# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="mlx_whisper_manager.py",
        description="MLX Whisper 模型管理工具（需先运行 fetch_mlx_model_catalog.py）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python3 fetch_mlx_model_catalog.py          # 先抓取模型目录\n"
            "  python3 mlx_whisper_manager.py list\n"
            "  python3 mlx_whisper_manager.py list --detail\n"
            "  python3 mlx_whisper_manager.py list --family small\n"
            "  python3 mlx_whisper_manager.py info whisper-tiny-mlx\n"
            "  python3 mlx_whisper_manager.py download whisper-tiny-mlx\n"
            "  python3 mlx_whisper_manager.py download whisper-tiny-mlx --output ./models\n"
            "  python3 mlx_whisper_manager.py download tiny-mlx           # 省略 whisper- 前缀\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── list 子命令 ──
    p_list = subparsers.add_parser("list", help="列出所有可用模型")
    p_list.add_argument(
        "--detail", action="store_true",
        help="显示详细信息（量化类型、内存需求、文件大小、下载状态）",
    )
    p_list.add_argument(
        "--family", metavar="FAMILY",
        help="按系列过滤（tiny / base / small / medium / large / turbo）",
    )
    p_list.add_argument(
        "--output", metavar="DIR",
        help=f"本地模型目录（用于检测已下载状态，默认：{DEFAULT_OUTPUT_DIR}）",
    )
    p_list.set_defaults(func=cmd_list)

    # ── info 子命令 ──
    p_info = subparsers.add_parser("info", help="显示指定模型的详细信息")
    p_info.add_argument("model_id", help="模型 ID（如 whisper-tiny-mlx 或 tiny-mlx）")
    p_info.add_argument(
        "--output", metavar="DIR",
        help=f"本地模型目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    p_info.set_defaults(func=cmd_info)

    # ── download 子命令 ──
    p_dl = subparsers.add_parser("download", help="下载指定模型的所有 Bundle 文件")
    p_dl.add_argument("model_id", help="模型 ID（如 whisper-tiny-mlx 或 tiny-mlx）")
    p_dl.add_argument(
        "--output", metavar="DIR",
        help=f"保存目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    p_dl.add_argument(
        "--force", action="store_true",
        help="强制重新下载（覆盖已存在的文件）",
    )
    p_dl.set_defaults(func=cmd_download)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
