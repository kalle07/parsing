"""Layout primitives and geometry helpers for PDF parsing.

This module defines a simple `BoundingBox` value object and a utility
function for filtering blocks by size when parsing PDFs.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple, Union


@dataclass
class BoundingBox:
    """Represents a 2‑dimensional axis‑aligned bounding box.

    The box is defined by its lower‑left (x0, y0) and upper‑right (x1, y1)
    corners.  All coordinates are assumed to be in the same unit system
(e.g. points from PDF content streams).

    Typical usage:
        >>> bbox = BoundingBox(10, 20, 100, 150)
        >>> bbox.width
        90.0
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Return the horizontal size of the box."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Return the vertical size of the box."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Return the total area of the box (width × height)."""
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """Return the x‑coordinate of the box's centre."""
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        """Return the y‑coordinate of the box's centre."""
        return (self.y0 + self.y1) / 2

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert the box to a 4‑tuple ``(x0, y0, x1, y1)``.

        The returned tuple can be fed directly into :meth:`from_tuple`.
        """
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_tuple(
        cls,
        bbox: Union[Tuple[float, float, float, float], List[float], "BoundingBox", Any],
    ) -> "BoundingBox":
        """Create a :class:`BoundingBox` from various input types.

        Supported inputs:
        * Another :class:`BoundingBox` instance (returned unchanged).
        * An object exposing ``x0``, ``y0``, ``x1`` and ``y1`` attributes
          (treated as a ``Rect``‑like object).
        * A 4‑element ``tuple`` or ``list`` of numbers.

        Parameters
         ----------
        bbox :
            The source data to convert.

        Returns
         -------
        BoundingBox
            A new bounding‑box instance.

        Raises
         ------
        ValueError
            If a tuple/list is supplied that does not contain exactly four
            coordinates.
        TypeError
            If the input is of an unsupported type.
        """
        # Fast path: already a BoundingBox – nothing to do.
        if isinstance(bbox, BoundingBox):
            return bbox

        # Objects that expose the four corner attributes (e.g. a custom Rect)
        if hasattr(bbox, "x0") and hasattr(bbox, "x1") and hasattr(bbox, "y0") and hasattr(bbox, "y1"):
            return cls(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

        # Sequence of four numbers (tuple or list)
        if isinstance(bbox, (tuple, list)):
            if len(bbox) != 4:
                raise ValueError(f"BoundingBox expects 4 coordinates, got {len(bbox)}")
            return cls(*bbox)

        # Anything else is unsupported
        raise TypeError(
            f"BoundingBox expects tuple, list, BoundingBox, or Rect, got {type(bbox).__name__}"
        )


def passes_size_filter(
    bbox: Union[BoundingBox, Tuple[float, float, float, float], List[float], Any],
    min_size_px: int = 10,
) -> bool:
    """Determine whether a block meets a minimum size requirement.

    The function converts the supplied ``bbox`` (which may be a
    :class:`BoundingBox`, a 4‑tuple, a list, or any object exposing the
    corner attributes) into a :class:`BoundingBox` instance, then computes
its width and height in pixels assuming a 72 dpi reference.  The
conversion factor ``/ 72 * 96`` translates points to screen pixels
(96 dpi).  The block passes if **either** dimension meets or exceeds
``min_size_px``.

    Parameters
     ----------
        bbox :
            The bounding box to test.
        min_size_px :
            Minimum dimension in pixels; defaults to 10.

        Returns
         -------
        bool
            ``True`` if the block satisfies the size filter, ``False`` otherwise.
        """
    # Normalise the input to a concrete BoundingBox instance.
    bbox_obj = bbox if isinstance(bbox, BoundingBox) else BoundingBox.from_tuple(bbox)

    # Convert from PDF points (72 dpi) to screen pixels (96 dpi).
    width_px = int(bbox_obj.width / 72 * 96)
    height_px = int(bbox_obj.height / 72 * 96)

    # Return True if either dimension meets the minimum size threshold.
    return width_px >= min_size_px or height_px >= min_size_px

