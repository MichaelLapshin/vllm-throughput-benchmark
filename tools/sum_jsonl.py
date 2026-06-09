"""
Tool for summing numerical values at a column of a JSONL file.
"""

import argparse
import json
import os
import sys


def sum_jsonl_column(file_path: str, column_name: str) -> float:
    """Reads a JSONL file line by line and sums the values of a specified column."""
    total_sum = 0.0
    line_number = 0

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line_number += 1
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                data = json.loads(line)
                # Check if the key exists in the current JSON object
                if column_name in data:
                    val = data[column_name]
                    # Ensure the value is a number before adding
                    if isinstance(val, (int, float)):
                        total_sum += val
                    elif isinstance(val, str):
                        total_sum += float(val)
                    elif val is None:
                        continue  # Safely ignore null values
                    else:
                        print(
                            f"Warning: Line {line_number} contains a non-numeric value '{val}' for column '{column_name}'. Skipping.",
                            file=sys.stderr,
                        )
            except json.JSONDecodeError:
                print(
                    f"Warning: Skipping invalid JSON on line {line_number}.",
                    file=sys.stderr,
                )

    return total_sum


def main():
    parser = argparse.ArgumentParser(
        description="Sum the values of a specific column in a JSONL file."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to the input JSONL file",
        metavar="FILE_PATH",
    )
    parser.add_argument(
        "-c",
        "--column",
        required=True,
        help="The column/key name to sum up",
        metavar="COLUMN_NAME",
    )

    args = parser.parse_args()

    result = sum_jsonl_column(args.file, args.column)

    # Format result as integer if it has no decimal part, otherwise float
    if result.is_integer():
        print(f"Total sum for '{args.column}': {int(result)}")
    else:
        print(f"Total sum for '{args.column}': {result}")


if __name__ == "__main__":
    main()