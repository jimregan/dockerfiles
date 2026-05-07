from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM


DEFAULT_MODEL_ID = "tiiuae/Falcon-OCR"


@dataclass
class RenderedPage:
    page_number: int
    image: Image.Image
    width: int
    height: int


@dataclass
class PageFailure:
    page_number: int
    stage: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Falcon OCR with layout on PDFs under /data and write page outputs to /output."
    )
    parser.add_argument("--input-root", type=Path, default=Path(os.environ.get("DATA_DIR", "/data")))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "/output")))
    parser.add_argument("--model-id", default=os.environ.get("FALCON_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--dpi", type=int, default=int(os.environ.get("RENDER_DPI", "200")))
    parser.add_argument(
        "--page-batch-size",
        type=int,
        default=int(os.environ.get("PAGE_BATCH_SIZE", "1")),
        help="How many rendered PDF pages to send to generate_with_layout() at once.",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=int(os.environ.get("OCR_BATCH_SIZE", "8")),
        help="Forwarded to Falcon OCR's generate_with_layout() for crop batching.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SKIP_EXISTING", True),
        help="Skip documents that already have a completed manifest.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=int(os.environ.get("MAX_DOCS", "0")),
        help="Process at most this many PDFs. 0 means no limit.",
    )
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=int(os.environ.get("MAX_PAGES_PER_DOC", "0")),
        help="Limit pages per document for debugging. 0 means no limit.",
    )
    return parser.parse_args()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model(model_id: str):
    dtype = select_dtype()
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if torch.cuda.is_available():
        device_index = int(os.environ.get("FALCON_CUDA_DEVICE", "0"))
        torch.cuda.set_device(device_index)
        kwargs["device_map"] = {"": device_index}

    print(
        f"Loading model {model_id} with dtype={dtype}"
        + (
            f" on cuda:{int(os.environ.get('FALCON_CUDA_DEVICE', '0'))} ..."
            if torch.cuda.is_available()
            else " on cpu ..."
        )
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    print("Model loaded.")
    return model


def find_pdfs(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    )


def output_dir_for_pdf(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_parent = pdf_path.relative_to(input_root).parent
    return output_root / relative_parent / pdf_path.stem


def render_pages(pdf_path: Path, dpi: int, max_pages: int) -> list[RenderedPage]:
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    rendered_pages: list[RenderedPage] = []

    with fitz.open(pdf_path) as document:
        total_pages = document.page_count
        if max_pages > 0:
            total_pages = min(total_pages, max_pages)

        for page_index in range(total_pages):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            rendered_pages.append(
                RenderedPage(
                    page_number=page_index + 1,
                    image=image,
                    width=image.width,
                    height=image.height,
                )
            )

    return rendered_pages


def batched(items: list[RenderedPage], batch_size: int) -> Iterable[list[RenderedPage]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def markdown_from_detections(detections: list[dict]) -> str:
    blocks: list[str] = []
    for detection in detections:
        text = (detection.get("text") or "").strip()
        if not text:
            continue

        category = detection.get("category", "text")
        if category == "title":
            blocks.append(f"# {text}")
        elif category in {"section-header", "page-header"}:
            blocks.append(f"## {text}")
        elif category == "list-item":
            cleaned = text.lstrip("-* ").strip()
            blocks.append(f"- {cleaned}")
        else:
            blocks.append(text)

    return "\n\n".join(blocks).strip() + "\n"


def serialize_detections(detections: list[dict]) -> list[dict]:
    serializable_detections = []
    for index, detection in enumerate(detections):
        serializable_detections.append(
            {
                "index": index,
                "category": detection.get("category"),
                "bbox": [int(value) for value in detection.get("bbox", [])],
                "score": float(detection.get("score", 0.0)),
                "text": detection.get("text", ""),
            }
        )
    return serializable_detections


def write_page_outputs(
    output_dir: Path,
    page_number: int,
    page_width: int,
    page_height: int,
    detections: list[dict],
) -> None:
    base_name = f"page-{page_number:04d}"
    json_path = output_dir / f"{base_name}.json"
    markdown_path = output_dir / f"{base_name}.md"

    json_payload = {
        "page": page_number,
        "width": page_width,
        "height": page_height,
        "blocks": detections,
    }

    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown_from_detections(detections), encoding="utf-8")


def write_page_error(output_dir: Path, page_number: int, stage: str, error: str) -> None:
    error_path = output_dir / f"page-{page_number:04d}.error.json"
    payload = {
        "page": page_number,
        "stage": stage,
        "error": error,
        "status": "failed",
    }
    error_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def maybe_clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cuda oom" in message
        or "cuda out of memory" in message
        or isinstance(exc, torch.cuda.OutOfMemoryError)
    )


def manifest_is_completed(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return manifest.get("status") == "completed"


def infer_layout_batch(model, pages: list[RenderedPage], ocr_batch_size: int) -> list[list[dict]]:
    images = [page.image for page in pages]
    return model.generate_with_layout(images, ocr_batch_size=ocr_batch_size)


def infer_layout_batch_resilient(
    model, pages: list[RenderedPage], ocr_batch_size: int
) -> tuple[list[list[dict]], list[PageFailure]]:
    try:
        return infer_layout_batch(model, pages, ocr_batch_size=ocr_batch_size), []
    except Exception as exc:  # noqa: BLE001
        maybe_clear_cuda_cache()

        if not is_cuda_oom_error(exc):
            raise

        page_numbers = [page.page_number for page in pages]
        print(
            f"CUDA OOM during layout inference for pages {page_numbers} "
            f"with ocr_batch_size={ocr_batch_size}; retrying with smaller work units."
        )

        if len(pages) > 1:
            midpoint = len(pages) // 2
            left_results, left_failures = infer_layout_batch_resilient(
                model, pages[:midpoint], ocr_batch_size=ocr_batch_size
            )
            right_results, right_failures = infer_layout_batch_resilient(
                model, pages[midpoint:], ocr_batch_size=ocr_batch_size
            )
            return left_results + right_results, left_failures + right_failures

        if ocr_batch_size > 1:
            smaller_ocr_batch_size = max(1, ocr_batch_size // 2)
            return infer_layout_batch_resilient(
                model, pages, ocr_batch_size=smaller_ocr_batch_size
            )

        return [], [
            PageFailure(
                page_number=pages[0].page_number,
                stage="layout_oom",
                error=str(exc),
            )
        ]


def infer_page_layout_only(model, page: RenderedPage, ocr_batch_size: int) -> list[dict]:
    results, failures = infer_layout_batch_resilient(
        model, [page], ocr_batch_size=ocr_batch_size
    )
    if failures:
        raise RuntimeError(failures[0].error)
    return serialize_detections(results[0])


def process_pdf(
    model,
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    dpi: int,
    page_batch_size: int,
    ocr_batch_size: int,
    max_pages_per_doc: int,
) -> dict:
    pdf_output_dir = output_dir_for_pdf(pdf_path, input_root, output_root)
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    rendered_pages = render_pages(pdf_path, dpi=dpi, max_pages=max_pages_per_doc)
    if not rendered_pages:
        return {
            "pdf": str(pdf_path.relative_to(input_root)),
            "output_dir": str(pdf_output_dir.relative_to(output_root)),
            "pages": 0,
            "status": "empty",
        }

    succeeded_pages = 0
    page_failures: list[PageFailure] = []

    for chunk in batched(rendered_pages, page_batch_size):
        try:
            results, chunk_failures = infer_layout_batch_resilient(
                model, chunk, ocr_batch_size=ocr_batch_size
            )
            failed_page_numbers = {failure.page_number for failure in chunk_failures}
            result_iter = iter(results)

            for page in chunk:
                if page.page_number in failed_page_numbers:
                    raise RuntimeError(
                        f"Layout inference could not fit page {page.page_number} in memory"
                    )

                detections = next(result_iter)
                serializable_detections = serialize_detections(detections)
                if not serializable_detections:
                    raise RuntimeError(
                        f"Layout inference returned no blocks for page {page.page_number}"
                    )
                write_page_outputs(
                    output_dir=pdf_output_dir,
                    page_number=page.page_number,
                    page_width=page.width,
                    page_height=page.height,
                    detections=serializable_detections,
                )
                succeeded_pages += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(
                f"Batch layout failed for {pdf_path.relative_to(input_root)} "
                f"pages {[page.page_number for page in chunk]}: {exc}"
            )
            maybe_clear_cuda_cache()

        for page in chunk:
            try:
                serializable_detections = infer_page_layout_only(
                    model, page, ocr_batch_size=ocr_batch_size
                )
                if not serializable_detections:
                    raise RuntimeError(
                        f"Layout inference returned no blocks for page {page.page_number}"
                    )
                write_page_outputs(
                    output_dir=pdf_output_dir,
                    page_number=page.page_number,
                    page_width=page.width,
                    page_height=page.height,
                    detections=serializable_detections,
                )
                succeeded_pages += 1
            except Exception as exc:  # noqa: BLE001
                maybe_clear_cuda_cache()
                page_failures.append(
                    PageFailure(page_number=page.page_number, stage="page_inference", error=str(exc))
                )
                write_page_error(
                    output_dir=pdf_output_dir,
                    page_number=page.page_number,
                    stage="page_inference",
                    error=str(exc),
                )

    status = "completed"
    if succeeded_pages == 0:
        status = "failed"
    elif page_failures:
        status = "partial"

    manifest = {
        "pdf": str(pdf_path.relative_to(input_root)),
        "output_dir": str(pdf_output_dir.relative_to(output_root)),
        "pages": len(rendered_pages),
        "succeeded_pages": succeeded_pages,
        "failed_pages": len(page_failures),
        "page_failures": [
            {"page": failure.page_number, "stage": failure.stage, "error": failure.error}
            for failure in page_failures
        ],
        "dpi": dpi,
        "model_id": getattr(model, "name_or_path", DEFAULT_MODEL_ID),
        "status": status,
    }
    (pdf_output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.input_root.exists():
        raise SystemExit(f"Input root does not exist: {args.input_root}")

    pdfs = find_pdfs(args.input_root)
    if args.max_docs > 0:
        pdfs = pdfs[: args.max_docs]

    print(f"Found {len(pdfs)} PDF files under {args.input_root}")
    if not pdfs:
        return

    model = load_model(args.model_id)

    completed = 0
    partial = 0
    skipped = 0
    failed = 0

    for pdf_path in tqdm(pdfs, desc="PDFs"):
        pdf_output_dir = output_dir_for_pdf(pdf_path, args.input_root, args.output_root)
        manifest_path = pdf_output_dir / "manifest.json"
        if args.skip_existing and manifest_is_completed(manifest_path):
            skipped += 1
            continue

        try:
            manifest = process_pdf(
                model=model,
                pdf_path=pdf_path,
                input_root=args.input_root,
                output_root=args.output_root,
                dpi=args.dpi,
                page_batch_size=args.page_batch_size,
                ocr_batch_size=args.ocr_batch_size,
                max_pages_per_doc=args.max_pages_per_doc,
            )
            if manifest["status"] == "completed":
                completed += 1
            elif manifest["status"] == "partial":
                partial += 1
            else:
                failed += 1
            print(
                f"Processed {manifest['pdf']} -> {manifest['output_dir']} "
                f"({manifest['succeeded_pages']}/{manifest['pages']} pages, {manifest['status']})"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            error_dir = pdf_output_dir
            error_dir.mkdir(parents=True, exist_ok=True)
            error_path = error_dir / "error.json"
            error_payload = {
                "pdf": str(pdf_path.relative_to(args.input_root)),
                "error": str(exc),
                "status": "failed",
            }
            error_path.write_text(json.dumps(error_payload, indent=2) + "\n", encoding="utf-8")
            print(f"Failed {pdf_path}: {exc}")

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "completed": completed,
        "partial": partial,
        "processed": completed + partial,
        "skipped": skipped,
        "failed": failed,
        "model_id": args.model_id,
        "dpi": args.dpi,
    }
    (args.output_root / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
