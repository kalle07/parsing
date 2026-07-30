"""Table extraction cleanup and structuring utilities."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from post_process import fix_hyphenated_lines, remove_nonprintable_chars


@dataclass
class TableData:
	"""Processed table data with metadata."""
	# Type of table: 'headline', 'standard', or 'no_data'
	type: str
	# Optional headline text (e.g., single-cell top row)
	headline: Optional[str]
	# Optional description text (e.g., single-cell bottom row)
	description: Optional[str]
	# List of column header names (cleaned and deduplicated)
	column_headers: List[str]
	# List of data rows, each as a dict mapping column headers to cell values
	data_rows: List[Dict[str, Any]]
	# Number of actual data rows (excluding header and empty rows)
	data_rows_count: int
	# Type of header structure: 'standard_headers' or 'corner_empty_table'
	header_structure_type: str
	# Whether the table has an empty top-left cell (i.e., corner is empty)
	is_corner_empty_table: bool
	# Whether the first row is recognized as a header row
	has_header_row: bool


class TableProcessor:
	"""Process and validate table data from PDF."""

	@staticmethod
	def clean_table_data(
		table_data: Union[List, Tuple],
		hyphen_fix_enabled: bool = True,
		return_fix_count: bool = False,
	) -> Union[List[Any], Tuple[List[Any], int]]:
		"""
		Clean all table cell values:
		- Remove non-printable characters
		- Optionally fix hyphenated line breaks (e.g., 'ex-\nample' -> 'example')
		- Preserve non-string values (e.g., numbers, None)
		"""
		cleaned = []
		fix_count = 0

		def clean_cell(cell: Any) -> Any:
			nonlocal fix_count
			if not isinstance(cell, str):
				return cell

			if hyphen_fix_enabled:
				cell, cell_fix_count = fix_hyphenated_lines(cell)
				fix_count += cell_fix_count

			return remove_nonprintable_chars(cell)

		for row in table_data:
			if isinstance(row, (list, tuple)):
				# Process each cell in the row
				cleaned_row = [clean_cell(cell) for cell in row]
				cleaned.append(cleaned_row)
			elif isinstance(row, str):
				# Single-cell row (e.g., malformed row)
				cleaned.append(clean_cell(row))
			else:
				# Non-string, non-list row: append as-is
				cleaned.append(row)

		if return_fix_count:
			return cleaned, fix_count
		return cleaned

	@staticmethod
	def is_useful_table(table_data: List[List]) -> bool:
		"""
		Validate if detected table contains useful structure.
		Filter out:
		- Empty tables
		- Tables with >75% empty rows
		- Tables with <3 meaningful cells
		- Single-row tables with multiple columns (likely headers only)
		"""
		if not table_data or len(table_data) == 0:
			return False

		# Count total cells
		total_cells = sum(1 for row in table_data for _ in row)
		if total_cells == 0:
			return False

		# Count rows that are entirely empty/placeholder
		empty_row_count = 0
		for row in table_data:
			if not isinstance(row, list):
				continue

			is_empty_row = True
			for cell in row:
				# Normalize cell to lowercase string for comparison
				cell_str = str(cell).strip().lower() if cell else ""
				if cell_str not in ("", "null", "none", "nan", "-", "_"):
					is_empty_row = False
					break
			if is_empty_row:
				empty_row_count += 1

		# Reject if >75% rows are empty
		if len(table_data) > 0 and empty_row_count / len(table_data) > 0.75:
			return False

		# Count non-empty, non-placeholder cells
		non_empty_cells = total_cells - sum(
			len([cell for cell in row if str(cell).strip().lower() in ("", "null", "none")])
			for row in table_data
		)
		if non_empty_cells < 3:
			return False

		# Reject single-row tables unless they have exactly one column (e.g., simple list)
		if len(table_data) == 1 and len(table_data[0]) > 1:
			return False

		return True

	@staticmethod
	def process_table(
		table_data: List[List[str]],
		page_num: int = None,
		table_num: int = None,
	) -> TableData:
		"""
		Process raw table data into structured TableData.

		Steps:
		1. Detect and extract optional headline/description (single-content rows at top/bottom).
		2. Determine if first row is header, and whether it's a corner-empty table.
		3. Clean and deduplicate headers.
		4. Process data rows accordingly (with or without corner-empty logic).
		"""
		# Handle empty input
		if not table_data or len(table_data) < 1:
			return TableData(
				type="no_data",
				headline=None,
				description=None,
				column_headers=[],
				data_rows=[],
				data_rows_count=0,
				header_structure_type="standard_headers",
				is_corner_empty_table=False,
				has_header_row=False,
			)

		# Initialize metadata
		headline_text = None
		description_text = None
		rows_for_processing = list(table_data)  # Copy to avoid mutation

		# Detect and extract headline (first row with exactly one content cell)
		if TableProcessor._has_single_content_cell(rows_for_processing[0]):
			headline_text = str(rows_for_processing[0][0]).strip()
			del rows_for_processing[0]  # Remove from processing

		# Detect and extract description (last row with exactly one content cell)
		if len(rows_for_processing) > 1 and TableProcessor._has_single_content_cell(rows_for_processing[-1]):
			description_text = str(rows_for_processing[-1][0]).strip()
			del rows_for_processing[-1]  # Remove from processing

		has_header_row = False
		header_structure_type = "standard_headers"
		headers: List[str] = []
		is_corner_empty = False

		# Try to detect header row (first remaining row)
		if len(rows_for_processing) > 0 and isinstance(rows_for_processing[0], list):
			potential_header_row = rows_for_processing[0]
			# Check if first cell is empty/placeholder → corner-empty table
			first_cell_empty = (
				len(potential_header_row) >= 2
				and str(potential_header_row[0]).strip() in ("", "null", "none", "-", "_")
			)

			if first_cell_empty:
				is_corner_empty = True
				header_structure_type = "corner_empty_table"
				has_header_row = True
				# Skip first cell; extract headers from remaining cells
				raw_headers = [
					str(cell).strip()
					for cell in potential_header_row[1:]
					if str(cell).strip() and str(cell).strip().lower() not in ("null", "none")
				]
			else:
				# Standard header row
				has_header_row = True
				raw_headers = [
					str(cell).strip()
					for cell in potential_header_row
					if str(cell).strip() and str(cell).strip().lower() not in ("null", "none")
				]

			# Deduplicate headers by appending suffixes (e.g., "Name", "Name_1")
			seen_headers: Dict[str, int] = {}
			for header in raw_headers:
				if header in seen_headers:
					seen_headers[header] += 1
					headers.append(f"{header}_{seen_headers[header]}")
				else:
					seen_headers[header] = 0
					headers.append(header)

		# Determine starting index for data rows
		data_rows_start_idx = 1 if has_header_row else 0
		data_rows = []

		# Process each data row
		for i in range(data_rows_start_idx, len(rows_for_processing)):
			current_row = rows_for_processing[i]
			if not isinstance(current_row, list):
				continue

			if is_corner_empty and len(headers) > 0:
				# Corner-empty table: first column is row labels (keys), rest are data
				row_label = str(current_row[0]).strip()
				# Skip rows with empty/placeholder labels
				if not row_label or row_label.lower() in ("null", "none"):
					continue

				# Build dict mapping column headers to cell values (skip label column)
				filtered_dict = {}
				for j, col_header in enumerate(headers):
					data_col_idx = 1 + j  # Skip first column (label)
					if data_col_idx < len(current_row):
						cell_val = str(current_row[data_col_idx]).strip()
						if cell_val and cell_val.lower() not in ("null", "none"):
							filtered_dict[col_header] = cell_val

				if filtered_dict:
					data_rows.append({row_label: filtered_dict})
			else:
				# Standard table: align cells directly with headers
				filtered_row = {}
				for j, col_header in enumerate(headers):
					if j < len(current_row):
						cell_val = str(current_row[j]).strip()
						if cell_val and cell_val.lower() not in ("null", "none"):
							filtered_row[col_header] = cell_val
				if filtered_row:
					data_rows.append(filtered_row)

		# Return structured TableData object
		return TableData(
			type="headline" if headline_text else "standard",
			headline=headline_text,
			description=description_text,
			column_headers=headers,
			data_rows=data_rows,
			data_rows_count=len(data_rows),
			header_structure_type=header_structure_type,
			is_corner_empty_table=is_corner_empty,
			has_header_row=has_header_row,
		)

	@staticmethod
	def _has_single_content_cell(row: List) -> bool:
		"""
		Check if row has exactly one cell with meaningful content.
		Empty, placeholder, or whitespace-only cells are excluded.
		"""
		if not isinstance(row, list):
			return False

		content_cells = [
			str(cell).strip()
			for cell in row
			if str(cell).strip() and str(cell).strip().lower() not in ("null", "none")
		]
		return len(content_cells) == 1
