"""Image extraction and persistence for PDF pages."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pymupdf

from layout import BoundingBox
from overlap import calculate_bbox_overlap, check_margin_violation


@dataclass
class ImageInfo:
    """Information about an extracted image."""

    img_idx: int
    xref: int
    page_num: int
    saved: bool
    filename: Optional[str]
    filepath: Optional[str]
    skipped_reason: Optional[str]
    width: int = 0
    height: int = 0
    original_format: str = "unknown"
    bbox: Optional[BoundingBox] = None


class ImageExtractor:
    """Extract and save images from PDF pages."""

    def __init__(self, config, pdf_filename: str, image_dir: str):
        """Initialize extractor with configuration and output directory."""
        self.config = config
        self.pdf_filename = pdf_filename
        self.image_dir = image_dir

    def extract_images(
        self,
        page: pymupdf.Page,
        page_num: int,
        text_blocks_bboxes: List[Tuple[float, float, float, float]] = None,
    ) -> Tuple[List[ImageInfo], List[Tuple[float, float, float, float]]]:
        """Extract images from a page.

        Returns:
            A tuple of:
                - ``image_list``: list of :class:`ImageInfo` objects for all images found.
                - ``image_bboxes``: list of bounding boxes that were actually saved.
        """
        image_list: List[ImageInfo] = []
        image_bboxes: List[Tuple[float, float, float, float]] = []
        page_image_count = 0

        # Retrieve raw image metadata from PyMuPDF.
        image_info_list = page.get_image_info(xrefs=True)
        if not image_info_list:
            return image_list, image_bboxes

        for img_idx, img_info in enumerate(image_info_list):
            xref = img_info.get("xref")
            img_bbox = img_info.get("bbox")
            img_width = img_info.get("width", 0)
            img_height = img_info.get("height", 0)

            # Process each image through validation, conflict resolution and optional saving.
            image_info = self._process_image(
                img_idx,
                xref,
                page_num,
                img_bbox,
                img_width,
                img_height,
                text_blocks_bboxes,
                page_image_count,
                page,
            )
            image_list.append(image_info)

            # If the image passed all checks and is to be saved, remember its bbox.
            if image_info.saved:
                image_bboxes.append(img_bbox)
                page_image_count += 1

        return image_list, image_bboxes

    def _process_image(
        self,
        img_idx: int,
        xref: int,
        page_num: int,
        img_bbox: Tuple[float, float, float, float],
        img_width: int,
        img_height: int,
        text_blocks_bboxes: Optional[List[Tuple[float, float, float, float]]],
        page_image_count: int,
        page: pymupdf.Page,
    ) -> ImageInfo:
        """Validate and possibly save a single image.

        All criteria are applied in the order defined by the original logic:
            1. Minimum size check.
            2. Full‑page image detection.
            3. Margin violation detection.
            4. Text‑block overlap detection.
            5. Maximum‑items‑per‑page limit.
            6. Final save decision based on configuration.
        """
        bbox = BoundingBox.from_tuple(img_bbox)
        min_dimension = min(img_width, img_height)

        # 1️⃣ Minimum dimension threshold – skip tiny images.
        if min_dimension < self.config.min_size_px:
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason=f"below {self.config.min_size_px}px threshold",
            )

        page_width = page.rect.width
        page_height = page.rect.height

        # 2️⃣ Full‑page image detection – skip images that cover most of the page.
        if page_width > 0 and page_height > 0:
            img_width_pts = bbox.width
            img_height_pts = bbox.height
            if img_width_pts >= 0.8 * page_width and img_height_pts >= 0.8 * page_height:
                return ImageInfo(
                    img_idx=img_idx + 1,
                    xref=xref,
                    page_num=page_num,
                    saved=False,
                    filename=None,
                    filepath=None,
                    skipped_reason=(
                        f"full-page image ({img_width_pts:.0f}x{img_height_pts:.0f}pts "
                        f"vs page {page_width:.0f}x{page_height:.0f}pts)"
                    ),
                )

        # 3️⃣ Margin violation check – images touching the outer margin are rejected.
        if check_margin_violation(bbox, page.rect.width, page.rect.height, self.config):
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason="Content lies on outer margin",
            )

        # 4️⃣ Overlap with nearby text blocks – reject if >90% overlap with similarly sized text.
        if text_blocks_bboxes and self._check_text_overlap(bbox, text_blocks_bboxes):
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason="overlaps >90% with similar-size text block",
            )

        # 5️⃣ Max‑items‑per‑page limit – stop processing once the limit is reached.
        if self.config.max_items_per_page > 0 and page_image_count >= self.config.max_items_per_page:
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason="max limit reached",
            )

        # 6️⃣ Final save decision – only performed when ``save_images`` is True.
        if self.config.save_images:
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=True,
                filename=f"{self.pdf_filename}_page_{page_num:04d}_img_{img_idx + 1:02d}.png",
                filepath=None,
                skipped_reason=None,
                width=img_width,
                height=img_height,
                original_format="png",
                bbox=bbox,
            )

        # If we reach here, the image is not saved for any of the above reasons.
        return ImageInfo(
            img_idx=img_idx + 1,
            xref=xref,
            page_num=page_num,
            saved=False,
            filename=None,
            filepath=None,
            skipped_reason="save_images=False",
            bbox=bbox,
        )

    def save_all(self, images: List[ImageInfo], doc: pymupdf.Document):
        """Write all successfully validated images to disk."""
        if any(img.saved for img in images):
            os.makedirs(self.image_dir, exist_ok=True)
        for img in images:
            if img.saved:
                img.filepath = os.path.join(self.image_dir, img.filename)
                self._save_image_to_disk(img, doc)

    def _save_image_to_disk(self, img: ImageInfo, doc: pymupdf.Document):
        """Perform the actual file I/O for a saved image."""
        try:
            # Load the image as a pixmap using its xref index.
            pix = pymupdf.Pixmap(doc, img.xref)
            png_data = pix.tobytes("png")
            # Write the binary PNG data to the target file.
            with open(img.filepath, "wb") as f:
                f.write(png_data)
            print(f"Saved image: {img.filename} ({img.width}x{img.height}px)")
        except Exception as exc:
            # If saving fails, mark the image as not saved and record the error.
            img.saved = False
            img.skipped_reason = f"Save failed: {str(exc)}"

    def _check_text_overlap(
        self,
        cluster_bbox: Union[Tuple[float, float, float, float], BoundingBox],
        reference_bboxes: List[Tuple[float, float, float, float]],
    ) -> bool:
        """Determine whether an image overlaps significantly with text blocks.

        An overlap is considered significant when:
            - The IoU (intersection‑over‑union) exceeds 0.9.
            - The areas of the two rectangles differ by less than 20 %.
        """
        bbox = (
            cluster_bbox
            if isinstance(cluster_bbox, BoundingBox)
            else BoundingBox.from_tuple(cluster_bbox)
        )
        for ref_bbox in reference_bboxes:
            ref_box = BoundingBox.from_tuple(ref_bbox)
            overlap_pct, _, has_overlap = calculate_bbox_overlap(bbox, ref_box)
            if has_overlap and overlap_pct >= 0.9:
                max_area = max(ref_box.area, bbox.area)
                size_diff = (
                    abs(ref_box.area - bbox.area) / max_area
                    if max_area > 0
                    else 1
                )
                if size_diff <= 0.2:
                    return True
        return False

