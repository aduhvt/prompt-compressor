from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.request
from typing import Callable

APP_VERSION = "1.0.0"
GITHUB_REPO = "aduhvt/prompt-compressor"
USER_AGENT = "PromptCompressor-Updater"


def parse_version(v_str: str) -> tuple[int, ...]:
    """Parse a version string (e.g. 'v1.0.2' -> (1, 0, 2)) safely."""
    v_str = v_str.lstrip('v')
    parts = []
    for part in v_str.split('.'):
        digits = ''.join(c for c in part if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def cleanup_old_executable() -> None:
    """Clean up the left-over renamed executable from a previous update."""
    try:
        current_exe = sys.executable
        old_exe = current_exe + ".old"
        if os.path.exists(old_exe):
            os.remove(old_exe)
    except Exception:
        # Silently ignore if the file is locked or cannot be deleted yet
        pass


class AppUpdater:
    def __init__(
        self,
        current_version: str = APP_VERSION,
        repo: str = GITHUB_REPO,
        on_update_available: Callable[[str, str], None] | None = None,
        on_progress: Callable[[float], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        self.current_version = current_version
        self.repo = repo
        self.on_update_available = on_update_available
        self.on_progress = on_progress
        self.on_error = on_error
        self.on_success = on_success
        self.download_url: str | None = None
        self.latest_version: str | None = None

    def start_check(self) -> None:
        """Start the update check in a background thread."""
        # Only run updater if packaged with PyInstaller
        if not getattr(sys, 'frozen', False):
            return
        threading.Thread(target=self._check_for_updates, daemon=True).start()

    def _check_for_updates(self) -> None:
        try:
            url = f"https://api.github.com/repos/{self.repo}/releases/latest"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            
            with urllib.request.urlopen(req, timeout=7) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            tag_name = data.get("tag_name", "")
            if not tag_name:
                return

            local_v = parse_version(self.current_version)
            remote_v = parse_version(tag_name)

            if remote_v > local_v:
                # Find the PromptCompressor.exe asset
                assets = data.get("assets", [])
                download_url = None
                for asset in assets:
                    if asset.get("name") == "PromptCompressor.exe":
                        download_url = asset.get("browser_download_url")
                        break
                
                if download_url:
                    self.download_url = download_url
                    self.latest_version = tag_name
                    if self.on_update_available:
                        self.on_update_available(tag_name, download_url)
        except Exception:
            # Silent failure for offline usage or network issues
            pass

    def start_download(self) -> None:
        """Start downloading the update in a background thread."""
        if not self.download_url:
            if self.on_error:
                self.on_error("No update URL available.")
            return
        threading.Thread(target=self._download_and_install, daemon=True).start()

    def _download_and_install(self) -> None:
        try:
            current_exe = sys.executable
            new_exe = current_exe + ".new"
            old_exe = current_exe + ".old"

            # Download the new binary
            req = urllib.request.Request(self.download_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as response:
                total_size = int(response.headers.get('content-length', 0))
                bytes_read = 0
                block_size = 16384
                
                with open(new_exe, 'wb') as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_read += len(chunk)
                        if total_size > 0 and self.on_progress:
                            progress = bytes_read / total_size
                            self.on_progress(progress)

            # Perform the safe swap:
            # 1. Rename running exe to .old
            if os.path.exists(old_exe):
                try:
                    os.remove(old_exe)
                except Exception:
                    pass
            
            os.rename(current_exe, old_exe)
            
            # 2. Rename downloaded .new to the original exe name
            os.rename(new_exe, current_exe)

            if self.on_success:
                self.on_success()

            # 3. Launch the new exe and terminate the old one
            subprocess.Popen([current_exe])
            os._exit(0)

        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
            # Clean up the .new file if it exists and failed
            try:
                current_exe = sys.executable
                new_exe = current_exe + ".new"
                if os.path.exists(new_exe):
                    os.remove(new_exe)
            except Exception:
                pass
