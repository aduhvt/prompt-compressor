from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk

# Add project root to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compressor.compress as compress_module
from compressor.similarity import DEFAULT_SIMILARITY_MODEL, calculate_similarity
from compressor.token_counter import compare_token_counts

# Types and Defaults
CompressionError = compress_module.CompressionError
compress_prompt = compress_module.compress_prompt
get_ollama_status = compress_module.get_ollama_status
DEFAULT_OLLAMA_MODEL = getattr(compress_module, "DEFAULT_OLLAMA_MODEL", "qwen3:1.7b")


class PromptCompressorApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # Configure Window
        self.title("Prompt Compressor")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Set theme colors
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")  # Themes: blue, green, dark-blue

        # Create Layout
        self.grid_columnconfigure(0, weight=0)  # Sidebar
        self.grid_columnconfigure(1, weight=1)  # Main Content
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_rowconfigure(7, weight=1)  # Spacer row

        # Sidebar Title
        self.title_label = ctk.CTkLabel(
            self.sidebar_frame, text="Prompt Compressor", font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame, text="Offline LLM Engine", font=ctk.CTkFont(size=12, slant="italic")
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Sidebar: Compression Mode
        self.mode_label = ctk.CTkLabel(self.sidebar_frame, text="Compression Mode:", anchor="w")
        self.mode_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.mode_option = ctk.CTkOptionMenu(
            self.sidebar_frame, values=["lossless", "balanced", "aggressive"]
        )
        self.mode_option.grid(row=3, column=0, padx=20, pady=(5, 15), sticky="ew")
        self.mode_option.set("balanced")

        # Sidebar: Ollama Model
        self.model_label = ctk.CTkLabel(self.sidebar_frame, text="Ollama Model:", anchor="w")
        self.model_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.model_entry = ctk.CTkEntry(self.sidebar_frame)
        self.model_entry.grid(row=5, column=0, padx=20, pady=(5, 10), sticky="ew")
        self.model_entry.insert(0, DEFAULT_OLLAMA_MODEL)

        # Sidebar: Check Ollama Status
        self.status_button = ctk.CTkButton(
            self.sidebar_frame, text="Refresh Ollama Status", command=self.check_ollama
        )
        self.status_button.grid(row=6, column=0, padx=20, pady=10, sticky="ew")

        # Status indicator label
        self.status_indicator = ctk.CTkLabel(
            self.sidebar_frame, text="Checking Ollama...", text_color="orange", anchor="w"
        )
        self.status_indicator.grid(row=7, column=0, padx=20, pady=5, sticky="nw")

        # Theme Selector
        self.appearance_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_label.grid(row=8, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.appearance_option = ctk.CTkOptionMenu(
            self.sidebar_frame, values=["System", "Dark", "Light"], command=self.change_appearance
        )
        self.appearance_option.grid(row=9, column=0, padx=20, pady=(5, 20), sticky="ew")

        # --- Main Frame ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=3)  # Input Textbox
        self.main_frame.grid_rowconfigure(3, weight=3)  # Output Textboxes
        self.main_frame.grid_rowconfigure(5, weight=0)  # Metrics Panel

        # Label: Input Prompt
        self.input_label = ctk.CTkLabel(
            self.main_frame, text="Input Prompt:", font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
        )
        self.input_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        # Input Text Area
        self.input_text = ctk.CTkTextbox(self.main_frame, height=180, font=("Consolas", 12))
        self.input_text.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        self.input_text.insert(
            "1.0",
            "Write a concise launch announcement for our offline prompt compression app. "
            "Mention that it runs locally with Ollama, measures token savings, and keeps meaning intact.",
        )

        # Compress Button / Progress Bar Frame
        self.control_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.control_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        self.control_frame.grid_columnconfigure(0, weight=1)

        self.compress_button = ctk.CTkButton(
            self.control_frame,
            text="Compress Prompt",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#0f766e",
            hover_color="#0d655e",
            height=40,
            command=self.start_compression,
        )
        self.compress_button.grid(row=0, column=0, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.control_frame, orientation="horizontal")
        self.progress_bar.configure(mode="indeterminate")

        self.status_lbl = ctk.CTkLabel(
            self.control_frame, text="Idle", font=ctk.CTkFont(size=12, slant="italic")
        )

        # Columns for Original vs Compressed Output
        # Column 0: Original Prompt
        self.original_label = ctk.CTkLabel(
            self.main_frame, text="Original Cleaned:", font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
        )
        self.original_label.grid(row=3, column=0, sticky="w", pady=(0, 5), padx=(0, 10))

        self.original_text = ctk.CTkTextbox(
            self.main_frame, height=200, font=("Consolas", 12), state="disabled"
        )
        self.original_text.grid(row=4, column=0, sticky="nsew", pady=(0, 5), padx=(0, 10))

        self.download_orig_btn = ctk.CTkButton(
            self.main_frame,
            text="Download Original",
            state="disabled",
            command=self.download_original,
        )
        self.download_orig_btn.grid(row=5, column=0, sticky="ew", pady=(0, 15), padx=(0, 10))

        # Column 1: Compressed Prompt
        self.compressed_label = ctk.CTkLabel(
            self.main_frame, text="Compressed:", font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
        )
        self.compressed_label.grid(row=3, column=1, sticky="w", pady=(0, 5), padx=(10, 0))

        self.compressed_text = ctk.CTkTextbox(
            self.main_frame, height=200, font=("Consolas", 12), state="disabled"
        )
        self.compressed_text.grid(row=4, column=1, sticky="nsew", pady=(0, 5), padx=(10, 0))

        # Action buttons for Compressed Output
        self.compressed_actions_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.compressed_actions_frame.grid(row=5, column=1, sticky="ew", pady=(0, 15), padx=(10, 0))
        self.compressed_actions_frame.grid_columnconfigure(0, weight=1)
        self.compressed_actions_frame.grid_columnconfigure(1, weight=1)

        self.copy_btn = ctk.CTkButton(
            self.compressed_actions_frame,
            text="Copy Compressed",
            state="disabled",
            fg_color="#111827",
            hover_color="#1f2937",
            command=self.copy_compressed,
        )
        self.copy_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.download_comp_btn = ctk.CTkButton(
            self.compressed_actions_frame,
            text="Download Compressed",
            state="disabled",
            command=self.download_compressed,
        )
        self.download_comp_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # --- Metrics Panel ---
        self.metrics_frame = ctk.CTkFrame(self.main_frame)
        self.metrics_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=5)
        self.metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.metric_orig_tokens = ctk.CTkLabel(
            self.metrics_frame, text="Original Tokens: -", font=ctk.CTkFont(size=13, weight="bold")
        )
        self.metric_orig_tokens.grid(row=0, column=0, pady=15)

        self.metric_comp_tokens = ctk.CTkLabel(
            self.metrics_frame, text="Compressed Tokens: -", font=ctk.CTkFont(size=13, weight="bold")
        )
        self.metric_comp_tokens.grid(row=0, column=1, pady=15)

        self.metric_reduction = ctk.CTkLabel(
            self.metrics_frame, text="Reduction: -", font=ctk.CTkFont(size=13, weight="bold")
        )
        self.metric_reduction.grid(row=0, column=2, pady=15)

        self.metric_similarity = ctk.CTkLabel(
            self.metrics_frame, text="Similarity: -", font=ctk.CTkFont(size=13, weight="bold")
        )
        self.metric_similarity.grid(row=0, column=3, pady=15)

        # Initial check
        self.last_result = None
        self.after(500, self.check_ollama)

    def change_appearance(self, new_mode: str) -> None:
        ctk.set_appearance_mode(new_mode)

    def check_ollama(self) -> None:
        def task():
            status = get_ollama_status()
            self.after(0, lambda: self.update_ollama_status(status))

        threading.Thread(target=task, daemon=True).start()

    def update_ollama_status(self, status: compress_module.OllamaStatus) -> None:
        model = self.model_entry.get().strip()
        if status.running:
            if status.available_models:
                models_str = ", ".join(status.available_models)
                if model in status.available_models:
                    self.status_indicator.configure(
                        text="Ollama: Connected (Model available)", text_color="green"
                    )
                else:
                    self.status_indicator.configure(
                        text=f"Ollama: Connected\n(Selected model '{model}' not found!\nAvailable: {models_str})",
                        text_color="orange",
                    )
            else:
                self.status_indicator.configure(
                    text="Ollama: Connected (No models installed)", text_color="orange"
                )
        elif status.installed:
            self.status_indicator.configure(
                text="Ollama: Local runtime not responding", text_color="red"
            )
        else:
            self.status_indicator.configure(
                text="Ollama: Not detected on system", text_color="red"
            )

    def start_compression(self) -> None:
        prompt_text = self.input_text.get("1.0", "end-1c").strip()
        if not prompt_text:
            messagebox.showwarning("Empty Input", "Please enter a prompt to compress.")
            return

        mode = self.mode_option.get()
        model = self.model_entry.get().strip()

        # Update UI state to loading
        self.compress_button.grid_forget()
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_bar.start()
        self.status_lbl.grid(row=1, column=0, pady=(5, 0))
        self.status_lbl.configure(text="Connecting to Ollama...")

        # Run compression in background thread
        threading.Thread(
            target=self.run_compression, args=(prompt_text, mode, model), daemon=True
        ).start()

    def run_compression(self, prompt_text: str, mode: str, model: str) -> None:
        start_time = time.perf_counter()
        try:
            self.after(0, lambda: self.status_lbl.configure(text="Compressing prompt using local Ollama model..."))
            result = compress_prompt(prompt_text, mode, model)

            self.after(0, lambda: self.status_lbl.configure(text="Calculating metrics and semantic similarity..."))
            token_stats = compare_token_counts(result.original_text, result.compressed_text)
            similarity = calculate_similarity(result.original_text, result.compressed_text)
            elapsed = time.perf_counter() - start_time

            self.after(
                0,
                lambda: self.on_compression_success(
                    result, token_stats, similarity, elapsed
                ),
            )
        except Exception as exc:
            self.after(0, lambda: self.on_compression_failure(exc))

    def on_compression_success(
        self,
        result: compress_module.CompressionResult,
        token_stats: any,
        similarity: any,
        elapsed: float,
    ) -> None:
        self.last_result = result

        # Reset buttons/loading states
        self.progress_bar.stop()
        self.progress_bar.grid_forget()
        self.status_lbl.grid_forget()
        self.compress_button.grid(row=0, column=0, sticky="ew")

        # Enable textareas, insert content, disable them again
        self.original_text.configure(state="normal")
        self.original_text.delete("1.0", "end")
        self.original_text.insert("1.0", result.original_text)
        self.original_text.configure(state="disabled")

        self.compressed_text.configure(state="normal")
        self.compressed_text.delete("1.0", "end")
        self.compressed_text.insert("1.0", result.compressed_text)
        self.compressed_text.configure(state="disabled")

        # Enable Action Buttons
        self.download_orig_btn.configure(state="normal")
        self.copy_btn.configure(state="normal")
        self.download_comp_btn.configure(state="normal")

        # Update Metrics
        self.metric_orig_tokens.configure(text=f"Original Tokens: {token_stats.original_tokens}")
        self.metric_comp_tokens.configure(text=f"Compressed Tokens: {token_stats.compressed_tokens}")
        self.metric_reduction.configure(text=f"Reduction: {token_stats.reduction_percentage:.1f}%")
        
        sim_suffix = ""
        if similarity.method != "sentence-transformers":
            sim_suffix = " (fallback)"
        self.metric_similarity.configure(text=f"Similarity: {similarity.score:.3f}{sim_suffix}")

        # Show notification
        self.bell()

    def on_compression_failure(self, error: Exception) -> None:
        # Reset buttons/loading states
        self.progress_bar.stop()
        self.progress_bar.grid_forget()
        self.status_lbl.grid_forget()
        self.compress_button.grid(row=0, column=0, sticky="ew")

        # Show error message
        error_msg = str(error)
        if isinstance(error, CompressionError):
            messagebox.showerror("Compression Error", f"Ollama Error:\n{error_msg}")
        else:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{error_msg}")

    def copy_compressed(self) -> None:
        if self.last_result:
            self.clipboard_clear()
            self.clipboard_append(self.last_result.compressed_text)
            self.update()  # keeps clipboard contents alive on Windows
            
            # Temporary change button text to indicate success
            old_text = self.copy_btn.cget("text")
            self.copy_btn.configure(text="Copied!")
            self.after(1500, lambda: self.copy_btn.configure(text=old_text))

    def download_original(self) -> None:
        if self.last_result:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                title="Save Original Prompt",
                initialfile="original-prompt.txt",
            )
            if filepath:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.last_result.original_text)

    def download_compressed(self) -> None:
        if self.last_result:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                title="Save Compressed Prompt",
                initialfile="compressed-prompt.txt",
            )
            if filepath:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.last_result.compressed_text)


if __name__ == "__main__":
    app = PromptCompressorApp()
    app.mainloop()
