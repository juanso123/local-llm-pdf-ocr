#!/usr/bin/env python3
"""
Local LLM PDF OCR - Command Line Interface

Process PDF documents through local LLM vision models to create searchable PDFs.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Rich imports (lightweight)
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def parse_page_range(page_str: str, total_pages: int) -> list:
    """
    Parse a page range string into a list of page numbers.
    
    Examples:
        "1-3" -> [0, 1, 2]
        "1,3,5" -> [0, 2, 4]
        "1-3,5,7-9" -> [0, 1, 2, 4, 6, 7, 8]
    
    Args:
        page_str: Comma-separated page ranges (1-indexed)
        total_pages: Total number of pages in the document
        
    Returns:
        List of 0-indexed page numbers
    """
    pages = set()
    for part in page_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            start = int(start.strip())
            end = int(end.strip())
            for p in range(start, end + 1):
                if 1 <= p <= total_pages:
                    pages.add(p - 1)  # Convert to 0-indexed
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                pages.add(p - 1)
    return sorted(pages)


def main():
    parser = argparse.ArgumentParser(
        description="PDF OCR with Local LLM - Transform scanned PDFs into searchable documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.pdf output_ocr.pdf
  %(prog)s input.pdf                      # Auto-generates input_ocr.pdf
  %(prog)s input.pdf output.pdf --verbose
  %(prog)s input.pdf output.pdf --pages 1-3,5
  %(prog)s input.pdf output.pdf --dpi 300 --api-base http://localhost:1234/v1
        """
    )
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument("output_pdf", nargs="?", help="Path to output PDF (default: <input>_ocr.pdf)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all output except errors")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for image rendering (default: 200)")
    parser.add_argument("--pages", help="Page range to process, e.g., '1-3,5' (default: all)")
    parser.add_argument("--api-base", help="Override LLM API base URL (default: from .env or localhost:1234)")
    parser.add_argument("--model", help="Override LLM model name (default: from .env or allenai/olmocr-2-7b)")
    
    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format='%(message)s', stream=sys.stderr)
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stderr)

    input_pdf_path = args.input_pdf
    
    # Generate output path if not provided
    if args.output_pdf:
        output_pdf_path = args.output_pdf
    else:
        input_path = Path(input_pdf_path)
        output_pdf_path = str(input_path.parent / f"{input_path.stem}_ocr.pdf")
    
    console = Console(quiet=args.quiet)
    
    # === LAZY IMPORTS ===
    # Import heavy modules (Surya, LLM) only after args are parsed
    # This makes --help fast since we skip model loading
    os.environ["TQDM_DISABLE"] = "1"  # Disable TQDM for Surya
    import base64
    from src.pdf_ocr.core.pdf import PDFHandler
    from src.pdf_ocr.core.ocr import OCRProcessor
    from src.pdf_ocr.core.aligner import HybridAligner
    
    # 1. Initialize Components
    pdf_handler = PDFHandler()
    ocr_processor = OCRProcessor(api_base=args.api_base, model=args.model)

    # 2. Convert PDF to Images
    console.print(f"[bold cyan]Processing '{input_pdf_path}'...[/bold cyan]")
    
    try:
        images_dict = pdf_handler.convert_to_images(input_pdf_path, dpi=args.dpi)
    except Exception as e:
        console.print(f"[bold red]Error converting PDF:[/bold red] {e}")
        sys.exit(1)
        
    page_nums = sorted(images_dict.keys())
    total_pages = len(page_nums)
    
    # Filter pages if --pages is specified
    if args.pages:
        selected_pages = parse_page_range(args.pages, total_pages)
        page_nums = [p for p in page_nums if p in selected_pages]
        console.print(f"[green]✓[/green] Processing [bold]{len(page_nums)}[/bold] of {total_pages} pages.")
    else:
        console.print(f"[green]✓[/green] Converted [bold]{total_pages}[/bold] pages.")

    pages_text = {}
    
    # 3. Hybrid OCR Loop with Rich Progress
    hybrid_aligner = HybridAligner()

    pages_structured_data = {}
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    )

    with progress:
        # ===== PHASE 1: Batch Layout Detection (Surya) =====
        task_layout = progress.add_task("[cyan]Detecting layouts (batch)...", total=1)
        
        # Collect all image bytes for batch processing
        all_image_bytes = []
        for page_num in page_nums:
            image_base64 = images_dict[page_num]
            image_bytes = base64.b64decode(image_base64)
            all_image_bytes.append(image_bytes)
        
        # Run batch detection (one Surya call for all pages)
        all_boxes = hybrid_aligner.get_detected_boxes_batch(all_image_bytes)
        progress.update(task_layout, completed=1)
        
        # ===== PHASE 2: LLM OCR + Alignment (per page) =====
        task_ocr = progress.add_task(f"[cyan]LLM OCR Processing ({len(page_nums)} pages)...", total=len(page_nums))
        
        for idx, page_num in enumerate(page_nums):
            progress.update(task_ocr, description=f"[cyan]LLM OCR Page {page_num + 1}/{total_pages}...")
            
            image_base64 = images_dict[page_num]
            boxes = all_boxes[idx]
            
            # Convert boxes to structured_data format
            structured_data = [(box, "") for box in boxes]
            
            # LLM OCR (Content)
            import asyncio
            llm_lines = asyncio.run(ocr_processor.perform_ocr(image_base64))
            logging.debug(f"DEBUG: LLM OCR retrieved {len(llm_lines)} lines for page {page_num}")
            
            # Align
            if llm_lines:
                aligned_data = hybrid_aligner.align_text(structured_data, llm_lines)
                pages_structured_data[page_num] = aligned_data
            else:
                pages_structured_data[page_num] = structured_data
            
            progress.advance(task_ocr)

    console.print("[green]✓[/green] OCR & Layout Analysis Complete.")

    # 4. Embed Text
    console.print("[bold blue]Embedding text into PDF...[/bold blue]")
    pdf_handler.embed_structured_text(input_pdf_path, output_pdf_path, pages_structured_data, dpi=args.dpi)

    console.print(f"[bold green]Done! Saved to '{output_pdf_path}'[/bold green]")


if __name__ == "__main__":
    main()
