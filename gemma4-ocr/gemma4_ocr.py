from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import fitz
from PIL import Image
from tqdm import tqdm


DEFAULT_MODEL_ID = "google/gemma-4-27b-it"
DEFAULT_OLLAMA_MODEL_ID = "gemma4:26b"

_OCR_PROMPT = (
    "Perform OCR on this document image with layout analysis. "
    "Return a JSON array where each element is a text block with: "
    '"text" (recognized text), '
    '"category" (one of: title, section-header, page-header, text, list-item, caption, table), '
    '"bbox" ([y1, x1, y2, x2] normalized to a 0-1000 grid). '
    "Return only the JSON array, no prose."
)

_FALLBACK_PROMPT = "Transcribe all text visible in this document image."


@dataclass
class RenderedPage:
    page_number: int
    image: Image.Image
    width: int
    height: int


class Backend(Protocol):
    model_id: str

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str: ...


@dataclass
class TransformersBackend:
    model_id: str
    _model: object
    _processor: object

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(new_ids, skip_special_tokens=True).strip()


@dataclass
class VLLMBackend:
    model_id: str
    _client: object

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content.strip()


@dataclass
class OllamaBackend:
    model_id: str
    _client: object

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        response = self._client.chat(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [buf.getvalue()],
                }
            ],
            options={"num_predict": max_new_tokens},
        )
        return response["message"]["content"].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 OCR with layout on PDFs under /data and write page outputs to /output."
    )
    parser.add_argument("--input-root", type=Path, default=Path(os.environ.get("DATA_DIR", "/data")))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "/output")))
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm", "ollama"],
        default=os.environ.get("BACKEND", "transformers"),
    )
    parser.add_argument("--model-id", default=None, help="Model ID; defaults vary by backend.")
    parser.add_argument("--dpi", type=int, default=int(os.environ.get("RENDER_DPI", "200")))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", "4096")),
        help="Maximum tokens to generate per page.",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.environ.get("VLLM_URL", "http://localhost:8000"),
        help="Base URL of the vLLM server (vllm backend only).",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Base URL of the Ollama server (ollama backend only).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SKIP_EXISTING", True),
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


def _default_model_id(backend: str) -> str:
    return DEFAULT_OLLAMA_MODEL_ID if backend == "ollama" else DEFAULT_MODEL_ID


def _select_dtype():
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_backend(backend: str, model_id: str, vllm_url: str, ollama_url: str) -> Backend:
    if backend == "transformers":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = _select_dtype()
        kwargs: dict = {"torch_dtype": dtype}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

        print(f"Loading model {model_id} with dtype={dtype} ...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        print("Model loaded.")
        return TransformersBackend(model_id=model_id, _model=model, _processor=processor)

    if backend == "vllm":
        from openai import OpenAI

        print(f"Connecting to vLLM at {vllm_url} ...")
        client = OpenAI(base_url=f"{vllm_url}/v1", api_key="vllm")
        return VLLMBackend(model_id=model_id, _client=client)

    if backend == "ollama":
        import ollama

        print(f"Connecting to Ollama at {ollama_url} ...")
        client = ollama.Client(host=ollama_url)
        return OllamaBackend(model_id=model_id, _client=client)

    raise ValueError(f"Unknown backend: {backend!r}")


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


def _parse_json_blocks(response: str) -> list[dict] | None:
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return None


def _normalize_bbox(bbox: list, width: int, height: int) -> list[int]:
    """Convert Gemma's [y1, x1, y2, x2] 0-1000 grid to [x1, y1, x2, y2] pixels."""
    y1, x1, y2, x2 = bbox
    return [
        int(x1 * width / 1000),
        int(y1 * height / 1000),
        int(x2 * width / 1000),
        int(y2 * height / 1000),
    ]


def ocr_page(backend: Backend, page: RenderedPage, max_new_tokens: int) -> list[dict]:
    response = backend.generate(page.image, _OCR_PROMPT, max_new_tokens)
    blocks = _parse_json_blocks(response)

    if blocks:
        result = []
        for index, block in enumerate(blocks):
            raw_bbox = block.get("bbox", [0, 0, 1000, 1000])
            result.append(
                {
                    "index": index,
                    "category": block.get("category", "text"),
                    "bbox": _normalize_bbox(raw_bbox, page.width, page.height),
                    "score": 1.0,
                    "text": (block.get("text") or "").strip(),
                }
            )
        return result

    fallback_text = backend.generate(page.image, _FALLBACK_PROMPT, max_new_tokens)
    return [
        {
            "index": 0,
            "category": "plain",
            "bbox": [0, 0, page.width, page.height],
            "score": 1.0,
            "text": fallback_text,
        }
    ]


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
            blocks.append(f"- {text.lstrip('-* ').strip()}")
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
    json_payload = {
        "page": page_number,
        "width": page_width,
        "height": page_height,
        "blocks": detections,
    }
    (output_dir / f"{base_name}.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / f"{base_name}.md").write_text(
        markdown_from_detections(detections), encoding="utf-8"
    )


def process_pdf(
    backend: Backend,
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    dpi: int,
    max_new_tokens: int,
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

    for page in rendered_pages:
        detections = ocr_page(backend, page, max_new_tokens)
        write_page_outputs(
            output_dir=pdf_output_dir,
            page_number=page.page_number,
            page_width=page.width,
            page_height=page.height,
            detections=detections,
        )

    manifest = {
        "pdf": str(pdf_path.relative_to(input_root)),
        "output_dir": str(pdf_output_dir.relative_to(output_root)),
        "pages": len(rendered_pages),
        "dpi": dpi,
        "model_id": backend.model_id,
        "status": "completed",
    }
    (pdf_output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.input_root.exists():
        raise SystemExit(f"Input root does not exist: {args.input_root}")

    model_id = args.model_id or _default_model_id(args.backend)
    backend = load_backend(args.backend, model_id, args.vllm_url, args.ollama_url)

    pdfs = find_pdfs(args.input_root)
    if args.max_docs > 0:
        pdfs = pdfs[: args.max_docs]

    print(f"Found {len(pdfs)} PDF files under {args.input_root}")
    if not pdfs:
        return

    processed = skipped = failed = 0

    for pdf_path in tqdm(pdfs, desc="PDFs"):
        pdf_output_dir = output_dir_for_pdf(pdf_path, args.input_root, args.output_root)
        if args.skip_existing and (pdf_output_dir / "manifest.json").exists():
            skipped += 1
            continue

        try:
            manifest = process_pdf(
                backend=backend,
                pdf_path=pdf_path,
                input_root=args.input_root,
                output_root=args.output_root,
                dpi=args.dpi,
                max_new_tokens=args.max_new_tokens,
                max_pages_per_doc=args.max_pages_per_doc,
            )
            processed += 1
            print(f"Processed {manifest['pdf']} -> {manifest['output_dir']} ({manifest['pages']} pages)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            (pdf_output_dir / "error.json").write_text(
                json.dumps(
                    {"pdf": str(pdf_path.relative_to(args.input_root)), "error": str(exc), "status": "failed"},
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            print(f"Failed {pdf_path}: {exc}")

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "model_id": backend.model_id,
        "dpi": args.dpi,
    }
    (args.output_root / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
