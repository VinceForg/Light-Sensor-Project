"""Append ESP32-S3 UV/IR readings to a CSV file, with a light-source label.

Each button press on the ESP32 sends one line over USB serial:
    uv,ir        (both scaled 0-1)
This script appends it as a row with the current label (no header row):
    uv,ir,label

Usage:
    pip install pyserial
    python csv_logger.py                         # COM3, writes to reading.csv
    python csv_logger.py COM3 my_data.csv

At startup you pick a label. While it runs, type a new label (or its number)
and press Enter to switch; every reading after that gets the new label.

Close the Arduino Serial Monitor first; only one program can use the port.
"""

import csv
import sys
import threading

import serial

DEFAULT_PORT = "COM3"
BAUD = 115200
LABELS = ["sun", "shade", "led", "incandescent"]

current_label = None


def parse_line(line):
    """Return [uv, ir] as strings, or None if the line isn't a data row."""
    fields = line.split(",")
    if len(fields) != 2:
        return None
    try:
        if not all(0.0 <= float(x) <= 1.0 for x in fields):
            return None
    except ValueError:
        return None
    return fields


def to_label(text):
    """Turn '1'-'4' into a preset label; anything else is used as typed."""
    text = text.strip().lower()
    if text.isdigit() and 1 <= int(text) <= len(LABELS):
        return LABELS[int(text) - 1]
    return text or None


def print_menu():
    print("Labels: " + "  ".join(f"{i}={name}" for i, name in enumerate(LABELS, 1)))
    print("(or type any other label)")


def ask_label():
    print_menu()
    label = None
    while label is None:
        label = to_label(input("Label: "))
    return label


def watch_keyboard():
    """Runs in the background so the label can change while readings come in."""
    global current_label
    while True:
        label = to_label(input())
        if label:
            current_label = label
            print(f"Label is now: {current_label}")


def main():
    global current_label
    port = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "figures/generated/reading.csv"

    current_label = ask_label()

    with serial.Serial(port, BAUD, timeout=1) as ser, \
            open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        threading.Thread(target=watch_keyboard, daemon=True).start()
        print(f"Logging {port} to {csv_path} with label '{current_label}'.")
        print("Type a new label + Enter to switch. Press Ctrl+C to stop.")

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue

            fields = parse_line(line)
            if fields is None:
                print(f"Skipped: {line}")
                continue

            writer.writerow(fields + [current_label])
            f.flush()   # save each row right away
            print(f"UV {fields[0]}, IR {fields[1]}, {current_label}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
