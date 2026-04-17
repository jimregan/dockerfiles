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
        default=int(os.environ.get("PAGE_BATCH_SIZE", "2")),
        help="How many rendered PDF pages to send to generate_with_layout() at once.",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=int(os.environ.get("OCR_BATCH_SIZE", "32")),
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
        kwargs["device_map"] = "auto"

    print(f"Loading model {model_id} with dtype={dtype} ...")
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

    for chunk in batched(rendered_pages, page_batch_size):
        images = [page.image for page in chunk]
        results = model.generate_with_layout(images, ocr_batch_size=ocr_batch_size)

        for page, detections in zip(chunk, results):
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

            if not serializable_detections:
                fallback_text = model.generate(page.image, category="plain")[0].strip()
                serializable_detections = [
                    {
                        "index": 0,
                        "category": "plain",
                        "bbox": [0, 0, page.width, page.height],
                        "score": 1.0,
                        "text": fallback_text,
                    }
                ]

            write_page_outputs(
                output_dir=pdf_output_dir,
                page_number=page.page_number,
                page_width=page.width,
                page_height=page.height,
                detections=serializable_detections,
            )

    manifest = {
        "pdf": str(pdf_path.relative_to(input_root)),
        "output_dir": str(pdf_output_dir.relative_to(output_root)),
        "pages": len(rendered_pages),
        "dpi": dpi,
        "model_id": getattr(model, "name_or_path", DEFAULT_MODEL_ID),
        "status": "completed",
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

    processed = 0
    skipped = 0
    failed = 0

    for pdf_path in tqdm(pdfs, desc="PDFs"):
        pdf_output_dir = output_dir_for_pdf(pdf_path, args.input_root, args.output_root)
        manifest_path = pdf_output_dir / "manifest.json"
        if args.skip_existing and manifest_path.exists():
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
            processed += 1
            print(
                f"Processed {manifest['pdf']} -> {manifest['output_dir']} "
                f"({manifest['pages']} pages)"
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
        "processed": processed,
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
