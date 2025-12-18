## yt_dlp_helper.py

Interactive wrapper around `yt-dlp` that helps list formats and download one or more YouTube videos or playlists.

### Requirements

- Python 3.8+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) available on your `PATH`

### Basic Usage

```bash
python yt_dlp_helper.py
```

Workflow:

1. Paste one or more URLs (video or playlist). Separate multiple entries with spaces or commas.
2. For each URL you can choose to inspect all available formats (`yt-dlp -F`).
3. Enter the format selector string (example: `137+140` or `bestvideo[height<=1080]+bestaudio`).
4. Optionally override the output template (defaults to `%(title)s.%(ext)s`).
5. Optionally provide extra `yt-dlp` flags (e.g., `--cookies /path/to/cookies.txt`).
6. The script downloads every URL sequentially using the provided settings.

### Tips

- Combine video and audio format codes with `+`, e.g., `137+140`.
- Use fallback expressions with `/`, e.g., `137+140/bestvideo[height=1080]+bestaudio/best`.
- Use format filters to pin resolutions: `bestvideo[height<=1080][fps<=60]+bestaudio`.
- For playlists, yt-dlp honors download archives, rate limits, etc., so feel free to pass extra flags via the prompt.
