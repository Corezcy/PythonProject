#!/usr/bin/env python3
"""
Interactive helper for downloading YouTube videos or playlists with yt-dlp.

Features:
* Accepts one or more URLs in a single prompt.
* Lets you inspect formats via `yt-dlp -F`.
* Prompts for a custom format selector (e.g., 137+140).
* Allows overriding the output template (-o).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import List


def ensure_yt_dlp_available() -> None:
    """Fail fast if yt-dlp is not on PATH."""
    if shutil.which("yt-dlp") is None:
        print("Error: yt-dlp executable not found on PATH.", file=sys.stderr)
        print("Install yt-dlp or ensure it is discoverable, then retry.", file=sys.stderr)
        sys.exit(1)


def prompt_urls() -> List[str]:
    """Prompt until at least one URL is provided."""
    while True:
        raw = input(
            "Enter one or more YouTube video/playlist URLs "
            "(separate multiple entries with spaces or commas):\n> "
        ).strip()
        urls = [chunk for chunk in re.split(r"[\s,]+", raw) if chunk]
        if urls:
            return urls
        print("No URLs detected. Please try again.")


def prompt_yes_no(message: str, default: bool = False) -> bool:
    """Generic yes/no prompt with a default choice."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        choice = input(f"{message} {suffix} ").strip().lower()
        if not choice:
            return default
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please respond with y or n.")


def prompt_format_selector() -> str:
    """Ask for the yt-dlp format selector string."""
    reminder = (
        "Format selector examples:\n"
        "  bestvideo[height<=1080]+bestaudio/best\n"
        '  137+140 (explicit video/audio combination)\n'
        "See yt-dlp docs for full syntax."
    )
    print(reminder)
    while True:
        fmt = input("Enter the format selector to use (-f): ").strip()
        if fmt:
            return fmt
        print("Format selector cannot be empty.")


def prompt_output_template() -> str:
    """Ask for output template, falling back to default."""
    default_template = "%(title)s.%(ext)s"
    template = input(
        f"Output template (-o). Press Enter to use default [{default_template}]: "
    ).strip()
    return template or default_template


def prompt_extra_args() -> List[str]:
    """Allow the user to pass through extra yt-dlp args."""
    extra = input(
        "Optional: enter extra yt-dlp arguments (e.g., --cookies /path/file). "
        "Leave blank to skip:\n> "
    ).strip()
    if not extra:
        return []
    # Basic splitting; users needing complex quoting should run yt-dlp manually.
    return extra.split()


def list_formats(url: str) -> None:
    """Invoke `yt-dlp -F` for the given URL."""
    print(f"\nListing formats for {url} ...\n")
    try:
        subprocess.run(["yt-dlp", "-F", url], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Failed to list formats for {url} (exit code {exc.returncode}).")


def download_url(
    url: str,
    format_selector: str,
    output_template: str,
    extra_args: List[str],
) -> None:
    """Invoke yt-dlp with the provided settings."""
    command = ["yt-dlp", "-f", format_selector, "-o", output_template, url, *extra_args]
    print(f"\nStarting download for {url} ...")
    print("Command:", " ".join(command))
    try:
        subprocess.run(command, check=True)
        print(f"Finished downloading {url}.\n")
    except subprocess.CalledProcessError as exc:
        print(f"yt-dlp failed for {url} (exit code {exc.returncode}).", file=sys.stderr)


def main() -> None:
    ensure_yt_dlp_available()
    urls = prompt_urls()

    for url in urls:
        if prompt_yes_no(f"Show available formats for {url}?", default=True):
            list_formats(url)

    fmt = prompt_format_selector()
    template = prompt_output_template()
    extra_args = prompt_extra_args()

    for url in urls:
        download_url(url, fmt, template, extra_args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
