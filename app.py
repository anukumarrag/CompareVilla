from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
SUPPORTED_FILE_TYPES = ["CSV", "XLS", "XLSX"]


def ensure_template_dir() -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_template_name(template_name: str) -> str:
    """Validate and normalize template filename input."""
    safe_name = template_name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", safe_name):
        raise ValueError("Template name may only contain letters, numbers, '_' or '-'.")
    return safe_name


def detect_file_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower().replace(".", "")
    if suffix == "xlsx":
        return "XLSX"
    if suffix == "xls":
        return "XLS"
    return "CSV"


def read_headers(uploaded_file: Any, file_type: str) -> list[str]:
    """Read only header columns from an uploaded CSV/XLS/XLSX file."""
    uploaded_file.seek(0)
    if file_type == "CSV":
        columns = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
    else:
        columns = pd.read_excel(uploaded_file, nrows=0).columns.tolist()
    uploaded_file.seek(0)
    return [str(col) for col in columns]


def read_dataframe(uploaded_file: Any, file_type: str) -> pd.DataFrame:
    """Read a full dataframe from an uploaded file and reset stream position."""
    uploaded_file.seek(0)
    if file_type == "CSV":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    uploaded_file.seek(0)
    return df


def template_files() -> list[Path]:
    ensure_template_dir()
    return sorted(TEMPLATE_DIR.glob("*.json"))


def save_template(template_name: str, payload: dict[str, Any]) -> Path:
    ensure_template_dir()
    safe_name = sanitize_template_name(template_name)

    template_root = TEMPLATE_DIR.resolve()
    file_name = f"{safe_name}.json"
    if Path(file_name).name != file_name:
        raise ValueError("Invalid template name.")
    output_path = (template_root / file_name).resolve()
    if output_path.parent != template_root:
        raise ValueError("Invalid template location.")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def load_template(template_path: Path) -> dict[str, Any]:
    with template_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_template_payload(
    template_name: str,
    primary_left: str,
    primary_right: str,
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build template payload persisted as JSON for later comparisons."""
    return {
        "templateName": template_name,
        "primaryColumn": {
            "left_file": primary_left,
            "right_file": primary_right,
            "type": "string",
        },
        "columnsToCompare": comparison_rows,
    }


def compare_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    template_config: dict[str, Any],
) -> pd.DataFrame:
    """Compare two dataframes based on configured key/column mappings."""
    primary = template_config["primaryColumn"]
    comparisons = template_config.get("columnsToCompare", [])

    left_key = primary["left_file"]
    right_key = primary["right_file"]

    left_prefixed = left_df.rename(columns=lambda c: f"left::{c}")
    right_prefixed = right_df.rename(columns=lambda c: f"right::{c}")

    merged = left_prefixed.merge(
        right_prefixed,
        how="outer",
        left_on=f"left::{left_key}",
        right_on=f"right::{right_key}",
        indicator=True,
    )

    merged["comparison_key"] = merged[f"left::{left_key}"].combine_first(
        merged[f"right::{right_key}"]
    )

    mismatch_any = pd.Series(False, index=merged.index)

    for mapping in comparisons:
        left_col = f"left::{mapping['left_file']}"
        right_col = f"right::{mapping['right_file']}"
        label = f"{mapping['left_file']} ↔ {mapping['right_file']}"

        if left_col not in merged.columns or right_col not in merged.columns:
            merged[f"{label} | mismatch"] = pd.Series(True, index=merged.index)
            mismatch_any = pd.Series(True, index=merged.index)
            continue

        if mapping.get("type", "string") == "numeric":
            threshold = float(mapping.get("threshold", 0))
            left_num = pd.to_numeric(merged[left_col], errors="coerce")
            right_num = pd.to_numeric(merged[right_col], errors="coerce")

            both_null = left_num.isna() & right_num.isna()
            both_present = left_num.notna() & right_num.notna()
            abs_diff = (left_num - right_num).abs()
            # Numeric mismatch when both values exist and exceed threshold,
            # or when one side is missing while the other has data.
            mismatch = (both_present & (abs_diff > threshold)) | (~both_present & ~both_null)
            merged[f"{label} | abs_diff"] = abs_diff
        else:
            left_raw = merged[left_col]
            right_raw = merged[right_col]
            both_null = left_raw.isna() & right_raw.isna()
            mismatch = (~both_null) & (
                left_raw.fillna("").astype(str) != right_raw.fillna("").astype(str)
            )

        mismatch = mismatch & (merged["_merge"] == "both")
        merged[f"{label} | mismatch"] = mismatch
        mismatch_any = mismatch_any | mismatch

    merged["row_status"] = np.where(
        merged["_merge"] == "left_only",
        "Missing in right",
        np.where(merged["_merge"] == "right_only", "Missing in left", "Matched"),
    )
    merged.loc[(merged["_merge"] == "both") & mismatch_any, "row_status"] = "Value mismatch"
    merged["has_discrepancy"] = merged["row_status"] != "Matched"

    priority_cols = {"comparison_key", "row_status", "has_discrepancy", "_merge"}
    ordered_cols = ["comparison_key", "row_status", "has_discrepancy", "_merge"] + [
        col for col in merged.columns if col not in priority_cols
    ]

    return merged[ordered_cols]


def export_dataframe_to_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="comparison_results")
    return buffer.getvalue()


def style_discrepancy_rows(row: pd.Series) -> list[str]:
    """Highlight rows that contain missing keys or field mismatches."""
    highlight = "background-color: #ffe6e6" if row.get("has_discrepancy", False) else ""
    return [highlight] * len(row)


def create_template_tab() -> None:
    st.subheader("Create Comparison Template")

    template_name = st.text_input("Template Name", key="create_template_name")
    selected_type = st.selectbox("File Type", SUPPORTED_FILE_TYPES, key="create_file_type")

    left_file = st.file_uploader(
        "Left File",
        type=["csv", "xls", "xlsx"],
        key="create_left_file",
    )
    right_file = st.file_uploader(
        "Right File",
        type=["csv", "xls", "xlsx"],
        key="create_right_file",
    )

    if not left_file or not right_file:
        st.info("Upload both files to configure mappings.")
        return

    try:
        left_headers = read_headers(left_file, selected_type)
        right_headers = read_headers(right_file, selected_type)
    except Exception as exc:
        st.error(f"Unable to parse file headers: {exc}")
        return

    if not left_headers or not right_headers:
        st.warning("No headers found in one of the files.")
        return

    st.markdown("### Primary Key Mapping")
    primary_left = st.selectbox("Primary Key (Left)", left_headers, key="primary_left")
    primary_right = st.selectbox("Primary Key (Right)", right_headers, key="primary_right")

    st.markdown("### Comparison Columns")
    row_count = st.number_input(
        "Number of column pairs",
        min_value=1,
        step=1,
        value=st.session_state.get("comparison_row_count", 1),
        key="comparison_row_count",
    )

    comparison_rows: list[dict[str, Any]] = []
    for idx in range(int(row_count)):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
        left_choice = c1.selectbox(
            f"Left Column #{idx + 1}",
            left_headers,
            key=f"compare_left_{idx}",
        )
        right_choice = c2.selectbox(
            f"Right Column #{idx + 1}",
            right_headers,
            key=f"compare_right_{idx}",
        )
        value_type = c3.selectbox(
            f"Type #{idx + 1}",
            ["string", "numeric"],
            key=f"compare_type_{idx}",
        )
        threshold = c4.number_input(
            f"Threshold #{idx + 1}",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key=f"compare_threshold_{idx}",
            disabled=value_type != "numeric",
        )

        row: dict[str, Any] = {
            "left_file": left_choice,
            "right_file": right_choice,
            "type": value_type,
        }
        if value_type == "numeric":
            row["threshold"] = threshold
        comparison_rows.append(row)

    if st.button("Save Template", key="save_template_button"):
        if not template_name.strip():
            st.error("Template Name is required.")
            return
        try:
            payload = build_template_payload(
                template_name=template_name.strip(),
                primary_left=primary_left,
                primary_right=primary_right,
                comparison_rows=comparison_rows,
            )
            output_path = save_template(template_name, payload)
            st.success(f"Template saved: {output_path.name}")
        except Exception as exc:
            st.error(f"Failed to save template: {exc}")


def compare_files_tab() -> None:
    st.subheader("Compare Files")
    templates = template_files()
    if not templates:
        st.info("No templates found. Create and save a template first.")
        return

    selected_template_name = st.selectbox(
        "Select Template",
        options=[tpl.name for tpl in templates],
        key="selected_template",
    )

    left_file = st.file_uploader(
        "Left File to Compare",
        type=["csv", "xls", "xlsx"],
        key="compare_left_file",
    )
    right_file = st.file_uploader(
        "Right File to Compare",
        type=["csv", "xls", "xlsx"],
        key="compare_right_file",
    )

    run_compare = st.button("Run Comparison", key="run_comparison")

    if run_compare:
        if not left_file or not right_file:
            st.error("Please upload both files for comparison.")
        else:
            selected_template = next(
                (tpl for tpl in templates if tpl.name == selected_template_name),
                None,
            )
            if selected_template is None:
                st.error("Selected template was not found.")
                return
            try:
                template_config = load_template(selected_template)
                left_df = read_dataframe(left_file, detect_file_type(left_file.name))
                right_df = read_dataframe(right_file, detect_file_type(right_file.name))
                st.session_state["comparison_results"] = compare_dataframes(
                    left_df,
                    right_df,
                    template_config,
                )
                st.session_state["comparison_template_used"] = selected_template_name
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    results_df = st.session_state.get("comparison_results")
    if results_df is None:
        return

    st.markdown("### Comparison Results")
    discrepancy_count = results_df["has_discrepancy"].sum()
    st.metric("Rows with discrepancy", discrepancy_count)

    styled = results_df.style.apply(style_discrepancy_rows, axis=1)
    st.dataframe(styled, use_container_width=True)

    excel_bytes = export_dataframe_to_excel(results_df)
    st.download_button(
        label="Download Results as Excel",
        data=excel_bytes,
        file_name="comparison_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_results",
    )


def main() -> None:
    st.set_page_config(page_title="CompareVillage", layout="wide")
    st.title("CompareVillage")

    create_tab, compare_tab = st.tabs(["Create Template", "Compare Files"])
    with create_tab:
        create_template_tab()
    with compare_tab:
        compare_files_tab()


if __name__ == "__main__":
    main()
