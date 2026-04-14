import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox


class AfroDanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AfroDance Learn")
        self.root.geometry("1380x860")
        self.root.minsize(1200, 760)
        self.root.configure(bg="#efe6d6")

        self.repo_root = Path(__file__).resolve().parent
        self.data_dir = self.repo_root / "data"
        self.references_dir = self.data_dir / "references"
        self.reference_json = self.references_dir / "instructor_reference.json"
        self.root_video = self.data_dir / "instructor.mp4"
        self.prototype_video = self.repo_root / "AfroDanceLearnPose" / "data" / "instructor.mp4"

        self.process_running = False

        self.build_ui()
        self.refresh_status(log_message=False)

    # ---------------- UI BUILD ----------------

    def build_ui(self):
        self.build_header()

        content = tk.Frame(self.root, bg="#efe6d6")
        content.pack(fill="both", expand=True, padx=18, pady=14)

        content.grid_columnconfigure(0, weight=0)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        self.build_left_sidebar(content)
        self.build_main_panels(content)

        self.build_footer()

    def build_header(self):
        header = tk.Frame(self.root, bg="#4f2f21", height=90)
        header.pack(fill="x")
        header.pack_propagate(False)

        title_frame = tk.Frame(header, bg="#4f2f21")
        title_frame.pack(side="left", padx=20, pady=12)

        tk.Label(
            title_frame,
            text="AfroDance Learn",
            font=("Arial", 24, "bold"),
            bg="#4f2f21",
            fg="#f8ebd3"
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text="Integrated Training Dashboard",
            font=("Arial", 12),
            bg="#4f2f21",
            fg="#ead7b6"
        ).pack(anchor="w", pady=(2, 0))

        self.header_status = tk.Frame(header, bg="#4f2f21")
        self.header_status.pack(side="right", padx=20, pady=16)

        self.video_badge = self.make_badge(self.header_status, "Video: Unknown")
        self.video_badge.pack(side="left", padx=6)

        self.reference_badge = self.make_badge(self.header_status, "Reference: Unknown")
        self.reference_badge.pack(side="left", padx=6)

        self.training_badge = self.make_badge(self.header_status, "Training: Unknown")
        self.training_badge.pack(side="left", padx=6)

    def make_badge(self, parent, text):
        return tk.Label(
            parent,
            text=text,
            font=("Arial", 10, "bold"),
            bg="#ead7b6",
            fg="#3a251a",
            padx=10,
            pady=6
        )

    def build_left_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg="#2f1f17", width=340, bd=2, relief="ridge")
        sidebar.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 18))
        sidebar.grid_propagate(False)

        tk.Label(
            sidebar,
            text="Main Workflow",
            font=("Arial", 18, "bold"),
            bg="#2f1f17",
            fg="#f6e6ca"
        ).pack(anchor="w", padx=18, pady=(18, 8))

        tk.Label(
            sidebar,
            text=(
                "Use these steps in order:\n"
                "1. Choose an instructor video\n"
                "2. Generate the reference data\n"
                "3. Start live dance training"
            ),
            font=("Arial", 11),
            justify="left",
            bg="#2f1f17",
            fg="#dbc6a8",
            wraplength=290
        ).pack(anchor="w", padx=18, pady=(0, 14))

        self.main_action_frame = tk.Frame(sidebar, bg="#2f1f17")
        self.main_action_frame.pack(fill="x", padx=18)

        self.btn_select_video = self.make_big_button(
            self.main_action_frame,
            "Select Instructor Video",
            "#1f6aa5",
            self.select_instructor_video
        )
        self.btn_select_video.pack(fill="x", pady=6)

        self.btn_copy_default = self.make_big_button(
            self.main_action_frame,
            "Use Included Prototype Video",
            "#6f42c1",
            self.copy_default_video
        )
        self.btn_copy_default.pack(fill="x", pady=6)

        self.btn_generate_reference = self.make_big_button(
            self.main_action_frame,
            "Generate Reference Data",
            "#2f9e44",
            lambda: self.run_script("extract_reference.py", "Reference generation")
        )
        self.btn_generate_reference.pack(fill="x", pady=6)

        self.btn_live_training = self.make_big_button(
            self.main_action_frame,
            "Start Live Training",
            "#198754",
            lambda: self.run_script("live_score.py", "Live training")
        )
        self.btn_live_training.pack(fill="x", pady=6)

        tk.Label(
            sidebar,
            text="Current Main Video",
            font=("Arial", 13, "bold"),
            bg="#2f1f17",
            fg="#f6e6ca"
        ).pack(anchor="w", padx=18, pady=(18, 6))

        self.video_info_label = tk.Label(
            sidebar,
            text="No root instructor video selected yet.",
            font=("Arial", 10),
            justify="left",
            bg="#3a2920",
            fg="#ebdcc3",
            wraplength=290,
            anchor="w",
            padx=10,
            pady=10,
            relief="sunken"
        )
        self.video_info_label.pack(fill="x", padx=18)

        tk.Label(
            sidebar,
            text="Project State",
            font=("Arial", 13, "bold"),
            bg="#2f1f17",
            fg="#f6e6ca"
        ).pack(anchor="w", padx=18, pady=(18, 6))

        self.project_state_label = tk.Label(
            sidebar,
            text="Idle",
            font=("Arial", 11, "bold"),
            bg="#b08968",
            fg="#1f130d",
            padx=10,
            pady=8
        )
        self.project_state_label.pack(fill="x", padx=18)

    def build_main_panels(self, parent):
        self.workflow_panel = self.create_panel(parent, "Primary User Flow", 0, 1)
        self.build_workflow_panel(self.workflow_panel)

        self.status_panel = self.create_panel(parent, "System Status & Log", 0, 2)
        self.build_status_panel(self.status_panel)

        self.advanced_panel = self.create_panel(parent, "Advanced Tools", 1, 1)
        self.build_advanced_panel(self.advanced_panel)

        self.files_panel = self.create_panel(parent, "Project Files & Readiness", 1, 2)
        self.build_files_panel(self.files_panel)

    def create_panel(self, parent, title, row, column):
        panel = tk.Frame(parent, bg="#6a4330", bd=2, relief="ridge")
        panel.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
        parent.grid_columnconfigure(column, weight=1)
        parent.grid_rowconfigure(row, weight=1)

        tk.Label(
            panel,
            text=title,
            font=("Arial", 17, "bold"),
            bg="#6a4330",
            fg="#f6e6ca"
        ).pack(anchor="w", padx=14, pady=(12, 8))

        inner = tk.Frame(panel, bg="#6a4330")
        inner.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return inner

    def build_workflow_panel(self, parent):
        rows = [
            ("Step 1", "Select an instructor video",
             "Choose your own video or use the included prototype video."),
            ("Step 2", "Generate reference data",
             "Extract pose frames and create the reference JSON used by training."),
            ("Step 3", "Start live training",
             "Open the webcam-based training mode with the instructor overlay and scoring."),
        ]

        for step, title, desc in rows:
            card = tk.Frame(parent, bg="#f3e7d5", bd=1, relief="solid")
            card.pack(fill="x", pady=8)

            tk.Label(
                card, text=step, font=("Arial", 10, "bold"),
                bg="#f3e7d5", fg="#7a4d35"
            ).pack(anchor="w", padx=12, pady=(10, 2))

            tk.Label(
                card, text=title, font=("Arial", 13, "bold"),
                bg="#f3e7d5", fg="#2f1f17"
            ).pack(anchor="w", padx=12)

            tk.Label(
                card, text=desc, font=("Arial", 10),
                bg="#f3e7d5", fg="#4d362b", wraplength=420, justify="left"
            ).pack(anchor="w", padx=12, pady=(4, 10))

    def build_status_panel(self, parent):
        top = tk.Frame(parent, bg="#6a4330")
        top.pack(fill="x", pady=(0, 8))

        tk.Button(
            top,
            text="Refresh Status",
            font=("Arial", 10, "bold"),
            bg="#6c757d",
            fg="white",
            command=self.manual_refresh_status
        ).pack(side="right")

        self.ready_summary = tk.Label(
            parent,
            text="Checking readiness...",
            font=("Arial", 11, "bold"),
            justify="left",
            bg="#6a4330",
            fg="#f6e6ca",
            anchor="w"
        )
        self.ready_summary.pack(fill="x", pady=(0, 10))

        self.log_text = tk.Text(
            parent,
            wrap="word",
            font=("Consolas", 10),
            bg="#f8f0e3",
            fg="#1e1e1e",
            height=18
        )
        self.log_text.pack(fill="both", expand=True)

    def build_advanced_panel(self, parent):
        tk.Label(
            parent,
            text=(
                "These tools are useful for testing, debugging, or examining the "
                "generated motion data. They are not the main user flow."
            ),
            font=("Arial", 10),
            justify="left",
            bg="#6a4330",
            fg="#f2e0c4",
            wraplength=500
        ).pack(anchor="w", pady=(0, 10))

        grid = tk.Frame(parent, bg="#6a4330")
        grid.pack(fill="both", expand=True)

        tools = [
            (
                "Reference Skeleton Viewer",
                "View the saved instructor skeleton frame by frame.",
                "#0d6efd",
                lambda: self.run_script("main.py", "Reference skeleton viewer")
            ),
            (
                "Advanced Overlay Analysis",
                "Open the overlay analysis mode for deeper pose comparison.",
                "#fd7e14",
                lambda: self.run_script("overlay.py", "Advanced overlay analysis")
            ),
        ]

        for title, desc, color, cmd in tools:
            card = tk.Frame(grid, bg="#f3e7d5", bd=1, relief="solid")
            card.pack(fill="x", pady=7)

            tk.Label(
                card,
                text=title,
                font=("Arial", 13, "bold"),
                bg="#f3e7d5",
                fg="#2f1f17"
            ).pack(anchor="w", padx=12, pady=(10, 2))

            tk.Label(
                card,
                text=desc,
                font=("Arial", 10),
                bg="#f3e7d5",
                fg="#4d362b",
                wraplength=500,
                justify="left"
            ).pack(anchor="w", padx=12, pady=(0, 8))

            tk.Button(
                card,
                text=f"Open {title}",
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                command=cmd
            ).pack(anchor="w", padx=12, pady=(0, 12))

    def build_files_panel(self, parent):
        self.files_text = tk.Label(
            parent,
            text="Checking files...",
            font=("Arial", 10),
            justify="left",
            anchor="nw",
            bg="#f3e7d5",
            fg="#2f1f17",
            padx=12,
            pady=12,
            relief="sunken"
        )
        self.files_text.pack(fill="both", expand=True)

    def build_footer(self):
        footer = tk.Frame(self.root, bg="#4f2f21", height=44)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        tk.Label(
            footer,
            text="Recommended use: Select video → Generate reference → Start live training",
            font=("Arial", 10),
            bg="#4f2f21",
            fg="#ead7b6"
        ).pack(side="left", padx=16, pady=10)

    def make_big_button(self, parent, text, bg, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 12, "bold"),
            bg=bg,
            fg="white",
            activebackground=bg,
            activeforeground="white",
            pady=12,
            relief="raised",
            bd=2
        )

    # ---------------- LOGGING / STATUS ----------------

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.root.update_idletasks()

    def set_project_state(self, text, bg="#b08968", fg="#1f130d"):
        self.project_state_label.config(text=text, bg=bg, fg=fg)

    def manual_refresh_status(self):
        self.refresh_status(log_message=True)
        self.set_project_state("Status refreshed", "#9ad0a0")

    def refresh_status(self, log_message=True):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(parents=True, exist_ok=True)

        video_ready = self.root_video.exists()
        reference_ready = self.reference_json.exists()
        training_ready = video_ready and reference_ready

        self.video_badge.config(
            text=f"Video: {'Ready' if video_ready else 'Missing'}",
            bg="#9ad0a0" if video_ready else "#e8b4b4"
        )
        self.reference_badge.config(
            text=f"Reference: {'Ready' if reference_ready else 'Missing'}",
            bg="#9ad0a0" if reference_ready else "#e8b4b4"
        )
        self.training_badge.config(
            text=f"Training: {'Ready' if training_ready else 'Not Ready'}",
            bg="#9ad0a0" if training_ready else "#e8c98c"
        )

        summary_lines = [
            f"Video file ready: {'Yes' if video_ready else 'No'}",
            f"Reference JSON ready: {'Yes' if reference_ready else 'No'}",
            f"Live training ready: {'Yes' if training_ready else 'No'}",
        ]
        self.ready_summary.config(text="\n".join(summary_lines))

        self.video_info_label.config(
            text=str(self.root_video) if video_ready else "No root instructor video selected yet."
        )

        files_text = (
            f"Root instructor video:\n"
            f"  {self.root_video}\n"
            f"  Exists: {'Yes' if video_ready else 'No'}\n\n"
            f"Reference JSON:\n"
            f"  {self.reference_json}\n"
            f"  Exists: {'Yes' if reference_ready else 'No'}\n\n"
            f"Prototype video:\n"
            f"  {self.prototype_video}\n"
            f"  Exists: {'Yes' if self.prototype_video.exists() else 'No'}\n\n"
            f"Available root scripts:\n"
            f"  extract_reference.py: {'Yes' if (self.repo_root / 'extract_reference.py').exists() else 'No'}\n"
            f"  live_score.py: {'Yes' if (self.repo_root / 'live_score.py').exists() else 'No'}\n"
            f"  main.py: {'Yes' if (self.repo_root / 'main.py').exists() else 'No'}\n"
            f"  overlay.py: {'Yes' if (self.repo_root / 'overlay.py').exists() else 'No'}"
        )
        self.files_text.config(text=files_text)

        self.btn_live_training.config(
            state=("normal" if training_ready else "disabled")
        )

        if log_message:
            self.log("Status refreshed successfully.")

    # ---------------- ACTIONS ----------------

    def select_instructor_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Instructor Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )

        if not file_path:
            return

        source = Path(file_path)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.root_video.write_bytes(source.read_bytes())
            self.log(f"Copied instructor video to {self.root_video}")
            self.refresh_status(log_message=False)
            self.set_project_state("Video selected", "#9ad0a0")
        except Exception as e:
            messagebox.showerror("Video Copy Error", f"Failed to copy selected video:\n{e}")
            self.log(f"ERROR copying selected video: {e}")
            self.set_project_state("Video copy failed", "#e8b4b4")

    def copy_default_video(self):
        source = None
        if self.prototype_video.exists():
            source = self.prototype_video
        elif self.root_video.exists():
            source = self.root_video

        if source is None:
            messagebox.showerror("Missing Video", "No default prototype instructor video was found.")
            self.log("ERROR: No prototype instructor video was found.")
            self.set_project_state("No prototype video found", "#e8b4b4")
            return

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.root_video.write_bytes(source.read_bytes())
            self.log(f"Copied default instructor video to {self.root_video}")
            self.refresh_status(log_message=False)
            self.set_project_state("Prototype video copied", "#9ad0a0")
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy prototype video:\n{e}")
            self.log(f"ERROR copying prototype video: {e}")
            self.set_project_state("Copy failed", "#e8b4b4")

    def run_script(self, script_name, label):
        if self.process_running:
            messagebox.showwarning("Busy", "Another project task is already running.")
            return

        script_path = self.repo_root / script_name
        if not script_path.exists():
            messagebox.showerror("Missing Script", f"Could not find:\n{script_path}")
            self.log(f"ERROR: Missing script {script_path}")
            return

        self.process_running = True
        self.set_project_state(f"Running: {label}", "#e8c98c")
        self.log(f"Launching {label}: {script_path}")

        thread = threading.Thread(
            target=self._run_script_thread,
            args=(script_path, label),
            daemon=True
        )
        thread.start()

    def _run_script_thread(self, script_path, label):
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True
            )
            self.root.after(0, lambda: self._handle_result(label, result))
        except Exception as e:
            self.root.after(0, lambda: self._handle_exception(label, e))

    def _handle_result(self, label, result):
        self.process_running = False
        self.log(f"--- {label} finished ---")

        if result.stdout.strip():
            self.log(result.stdout.strip())
        if result.stderr.strip():
            self.log(result.stderr.strip())

        if result.returncode == 0:
            self.set_project_state(f"Completed: {label}", "#9ad0a0")
        else:
            self.set_project_state(f"Issue in: {label}", "#e8b4b4")
            messagebox.showwarning(
                "Finished With Issues",
                f"{label} finished with return code {result.returncode}.\nCheck the log panel for details."
            )

        self.refresh_status(log_message=False)

    def _handle_exception(self, label, exception):
        self.process_running = False
        self.log(f"ERROR running {label}: {exception}")
        self.set_project_state(f"Execution failed: {label}", "#e8b4b4")
        messagebox.showerror("Execution Error", f"Failed to run {label}:\n{exception}")
        self.refresh_status(log_message=False)


if __name__ == "__main__":
    root = tk.Tk()
    app = AfroDanceApp(root)
    root.mainloop()
