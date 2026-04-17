import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from dance_library import (
    copy_into_selected_dance,
    copy_prototype_to_selected,
    ensure_library_structure,
    get_selected_dance,
    list_dances,
    prepare_selected_dance_for_extraction,
    save_selected_dance,
    store_generated_reference_for_selected,
)


class AfroDanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AfroDance Learn")
        self.root.geometry("1420x900")
        self.root.minsize(1220, 780)
        self.root.configure(bg="#efe6d6")

        self.repo_root = Path(__file__).resolve().parent
        self.data_dir = self.repo_root / "data"
        self.references_dir = self.data_dir / "references"
        self.root_video = self.data_dir / "instructor.mp4"
        self.generated_reference = self.references_dir / "instructor_reference.json"

        ensure_library_structure(self.repo_root)

        self.process_running = False
        self.dance_var = tk.StringVar()
        self.dance_name_to_id = {}

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
            fg="#f8ebd3",
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text="Dance Library + Live Training Dashboard",
            font=("Arial", 12),
            bg="#4f2f21",
            fg="#ead7b6",
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
            pady=6,
        )

    def build_left_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg="#2f1f17", width=360, bd=2, relief="ridge")
        sidebar.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 18))
        sidebar.grid_propagate(False)

        tk.Label(
            sidebar,
            text="Dance Library",
            font=("Arial", 18, "bold"),
            bg="#2f1f17",
            fg="#f6e6ca",
        ).pack(anchor="w", padx=18, pady=(18, 8))

        tk.Label(
            sidebar,
            text=(
                "Select a dance included in the project, generate or refresh its reference data, "
                "and use that dance in Live Training, Detailed Analysis Mode, and the Reference Pose Viewer."
            ),
            font=("Arial", 10),
            justify="left",
            bg="#2f1f17",
            fg="#dbc6a8",
            wraplength=310,
        ).pack(anchor="w", padx=18, pady=(0, 12))

        tk.Label(
            sidebar,
            text="Selected Dance",
            font=("Arial", 13, "bold"),
            bg="#2f1f17",
            fg="#f6e6ca",
        ).pack(anchor="w", padx=18, pady=(8, 4))

        self.dance_menu = tk.OptionMenu(sidebar, self.dance_var, "")
        self.dance_menu.config(font=("Arial", 11), bg="#ead7b6", fg="#2f1f17", width=28)
        self.dance_menu.pack(fill="x", padx=18, pady=(0, 10))

        self.dance_meta_label = tk.Label(
            sidebar,
            text="No dances found in data/dances.",
            font=("Arial", 10),
            justify="left",
            bg="#3a2920",
            fg="#ebdcc3",
            wraplength=310,
            anchor="w",
            padx=10,
            pady=10,
            relief="sunken",
        )
        self.dance_meta_label.pack(fill="x", padx=18)

        tk.Label(
            sidebar,
            text="Selected Dance Actions",
            font=("Arial", 13, "bold"),
            bg="#2f1f17",
            fg="#f6e6ca",
        ).pack(anchor="w", padx=18, pady=(18, 6))

        action_frame = tk.Frame(sidebar, bg="#2f1f17")
        action_frame.pack(fill="x", padx=18)

        self.btn_import_video = self.make_big_button(
            action_frame,
            "Import Video to Selected Dance",
            "#1f6aa5",
            self.import_video_to_selected_dance,
        )
        self.btn_import_video.pack(fill="x", pady=6)

        self.btn_copy_prototype = self.make_big_button(
            action_frame,
            "Use Included Prototype for Selected Dance",
            "#6f42c1",
            self.copy_prototype_to_selected_dance,
        )
        self.btn_copy_prototype.pack(fill="x", pady=6)

        self.btn_generate_reference = self.make_big_button(
            action_frame,
            "Generate Reference for Selected Dance",
            "#2f9e44",
            self.generate_reference_for_selected_dance,
        )
        self.btn_generate_reference.pack(fill="x", pady=6)

        self.btn_live_training = self.make_big_button(
            action_frame,
            "Start Live Training",
            "#198754",
            lambda: self.run_script("live_score.py", "Live training"),
        )
        self.btn_live_training.pack(fill="x", pady=6)

        tk.Label(
            sidebar,
            text="Project State",
            font=("Arial", 13, "bold"),
            bg="#2f1f17",
            fg="#f6e6ca",
        ).pack(anchor="w", padx=18, pady=(18, 6))

        self.project_state_label = tk.Label(
            sidebar,
            text="Idle",
            font=("Arial", 11, "bold"),
            bg="#b08968",
            fg="#1f130d",
            padx=10,
            pady=8,
        )
        self.project_state_label.pack(fill="x", padx=18)

    def build_main_panels(self, parent):
        self.workflow_panel = self.create_panel(parent, "Primary User Flow", 0, 1)
        self.build_workflow_panel(self.workflow_panel)

        self.status_panel = self.create_panel(parent, "System Status & Log", 0, 2)
        self.build_status_panel(self.status_panel)

        self.advanced_panel = self.create_panel(parent, "Advanced Tools", 1, 1)
        self.build_advanced_panel(self.advanced_panel)

        self.files_panel = self.create_panel(parent, "Dance Library Status", 1, 2)
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
            fg="#f6e6ca",
        ).pack(anchor="w", padx=14, pady=(12, 8))

        inner = tk.Frame(panel, bg="#6a4330")
        inner.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return inner

    def build_workflow_panel(self, parent):
        rows = [
            (
                "Step 1",
                "Select a dance from the library",
                "Choose one of the included dance options in the left panel.",
            ),
            (
                "Step 2",
                "Import or update the dance video",
                "You can copy the included prototype into the selected dance folder or import your own video.",
            ),
            (
                "Step 3",
                "Generate the selected dance reference",
                "This creates the selected dance's reference.json file.",
            ),
            (
                "Step 4",
                "Run training or analysis",
                "Start Live Training, Detailed Analysis Mode, or Reference Pose Viewer using the selected dance.",
            ),
        ]

        for step, title, desc in rows:
            card = tk.Frame(parent, bg="#f3e7d5", bd=1, relief="solid")
            card.pack(fill="x", pady=8)

            tk.Label(
                card,
                text=step,
                font=("Arial", 10, "bold"),
                bg="#f3e7d5",
                fg="#7a4d35",
            ).pack(anchor="w", padx=12, pady=(10, 2))

            tk.Label(
                card,
                text=title,
                font=("Arial", 13, "bold"),
                bg="#f3e7d5",
                fg="#2f1f17",
            ).pack(anchor="w", padx=12)

            tk.Label(
                card,
                text=desc,
                font=("Arial", 10),
                bg="#f3e7d5",
                fg="#4d362b",
                wraplength=420,
                justify="left",
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
            command=self.manual_refresh_status,
        ).pack(side="right")

        self.ready_summary = tk.Label(
            parent,
            text="Checking readiness...",
            font=("Arial", 11, "bold"),
            justify="left",
            bg="#6a4330",
            fg="#f6e6ca",
            anchor="w",
        )
        self.ready_summary.pack(fill="x", pady=(0, 10))

        self.log_text = tk.Text(
            parent,
            wrap="word",
            font=("Consolas", 10),
            bg="#f8f0e3",
            fg="#1e1e1e",
            height=18,
        )
        self.log_text.pack(fill="both", expand=True)

    def build_advanced_panel(self, parent):
        tk.Label(
            parent,
            text=(
                "These tools use the currently selected dance from the dance library."
            ),
            font=("Arial", 10),
            justify="left",
            bg="#6a4330",
            fg="#f2e0c4",
            wraplength=500,
        ).pack(anchor="w", pady=(0, 10))

        tools = [
            (
                "Reference Pose Viewer",
                "View the selected dance reference skeleton frame by frame.",
                "#0d6efd",
                lambda: self.run_script("main.py", "Reference pose viewer"),
            ),
            (
                "Detailed Analysis Mode",
                "Open the technical comparison mode for the selected dance.",
                "#fd7e14",
                lambda: self.run_script("overlay.py", "Detailed analysis mode"),
            ),
        ]

        for title, desc, color, cmd in tools:
            card = tk.Frame(parent, bg="#f3e7d5", bd=1, relief="solid")
            card.pack(fill="x", pady=7)

            tk.Label(
                card,
                text=title,
                font=("Arial", 13, "bold"),
                bg="#f3e7d5",
                fg="#2f1f17",
            ).pack(anchor="w", padx=12, pady=(10, 2))

            tk.Label(
                card,
                text=desc,
                font=("Arial", 10),
                bg="#f3e7d5",
                fg="#4d362b",
                wraplength=500,
                justify="left",
            ).pack(anchor="w", padx=12, pady=(0, 8))

            tk.Button(
                card,
                text=f"Open {title}",
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                command=cmd,
            ).pack(anchor="w", padx=12, pady=(0, 12))

    def build_files_panel(self, parent):
        container = tk.Frame(parent, bg="#f3e7d5", relief="sunken", bd=1)
        container.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(container)
        scrollbar.pack(side="right", fill="y")

        self.files_text = tk.Text(
            container,
            font=("Arial", 10),
            bg="#f3e7d5",
            fg="#2f1f17",
            wrap="word",
            yscrollcommand=scrollbar.set,
            padx=12,
            pady=12,
            relief="flat"
        )
        self.files_text.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=self.files_text.yview)

        self.files_text.insert("1.0", "Checking dance library...")
        self.files_text.config(state="disabled")

    def build_footer(self):
        footer = tk.Frame(self.root, bg="#4f2f21", height=44)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        tk.Label(
            footer,
            text="Recommended use: Select dance → Generate selected reference → Start Live Training",
            font=("Arial", 10),
            bg="#4f2f21",
            fg="#ead7b6",
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
            bd=2,
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

    def refresh_dance_menu(self):
        dances = list_dances(self.repo_root)
        menu = self.dance_menu["menu"]
        menu.delete(0, "end")
        self.dance_name_to_id = {}

        if not dances:
            self.dance_var.set("No dances found")
            menu.add_command(label="No dances found", command=lambda: None)
            return

        current = get_selected_dance(self.repo_root)
        selected_name = current["name"] if current else dances[0]["name"]

        for dance in dances:
            self.dance_name_to_id[dance["name"]] = dance["id"]
            menu.add_command(
                label=dance["name"],
                command=lambda value=dance["name"]: self.on_dance_selected(value),
            )

        self.dance_var.set(selected_name)

    def on_dance_selected(self, dance_name):
        self.dance_var.set(dance_name)
        dance_id = self.dance_name_to_id.get(dance_name)
        if dance_id:
            save_selected_dance(self.repo_root, dance_id)
        self.refresh_status(log_message=False)

    def refresh_status(self, log_message=True):
        ensure_library_structure(self.repo_root)

        self.refresh_dance_menu()
        selected = get_selected_dance(self.repo_root)
        dances = list_dances(self.repo_root)

        if selected is None:
            self.video_badge.config(text="Video: Missing", bg="#e8b4b4")
            self.reference_badge.config(text="Reference: Missing", bg="#e8b4b4")
            self.training_badge.config(text="Training: Not Ready", bg="#e8c98c")
            self.ready_summary.config(text="No dances found in data/dances.")
            self.dance_meta_label.config(text="No dances found in data/dances.")
            self.files_text.config(text="No dances found in data/dances.")
            self.btn_generate_reference.config(state="disabled")
            self.btn_live_training.config(state="disabled")
            return

        video_ready = selected["video_exists"]
        reference_ready = selected["reference_exists"]
        training_ready = video_ready and reference_ready

        self.video_badge.config(
            text=f"Video: {'Ready' if video_ready else 'Missing'}",
            bg="#9ad0a0" if video_ready else "#e8b4b4",
        )
        self.reference_badge.config(
            text=f"Reference: {'Ready' if reference_ready else 'Missing'}",
            bg="#9ad0a0" if reference_ready else "#e8b4b4",
        )
        self.training_badge.config(
            text=f"Training: {'Ready' if training_ready else 'Not Ready'}",
            bg="#9ad0a0" if training_ready else "#e8c98c",
        )

        self.ready_summary.config(
            text=(
                f"Selected Dance: {selected['name']}\n"
                f"Video file ready: {'Yes' if video_ready else 'No'}\n"
                f"Reference JSON ready: {'Yes' if reference_ready else 'No'}\n"
                f"Live training ready: {'Yes' if training_ready else 'No'}"
            )
        )

        self.dance_meta_label.config(
            text=(
                f"Name: {selected['name']}\n"
                f"Region: {selected.get('region', 'Unknown')}\n"
                f"Difficulty: {selected.get('difficulty', 'Unknown')}\n\n"
                f"{selected.get('description', 'No description available.')}\n\n"
                f"Folder: {selected['folder']}"
            )
        )

        lines = ["Dance Library Files:"]
        for dance in dances:
            lines.append(f"\n{dance['name']} ({dance['id']})")
            lines.append(f"  Video: {'Yes' if dance['video_exists'] else 'No'}")
            lines.append(f"  Reference: {'Yes' if dance['reference_exists'] else 'No'}")
            lines.append(f"  Ready: {'Yes' if dance['ready'] else 'No'}")
        self.files_text.config(state="normal")
        self.files_text.delete("1.0", "end")
        self.files_text.insert("1.0", "\n".join(lines))
        self.files_text.config(state="disabled")

        self.btn_generate_reference.config(state="normal" if video_ready else "disabled")
        self.btn_live_training.config(state="normal" if training_ready else "disabled")

        if log_message:
            self.log("Status refreshed successfully.")

    # ---------------- ACTIONS ----------------

    def import_video_to_selected_dance(self):
        selected = get_selected_dance(self.repo_root)
        if selected is None:
            messagebox.showerror("No Dance Selected", "No dance folders were found in data/dances.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Instructor Video for Selected Dance",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )

        if not file_path:
            return

        source = Path(file_path)

        try:
            copy_into_selected_dance(self.repo_root, source)
            self.log(f"Copied video into selected dance: {selected['id']}")
            self.refresh_status(log_message=False)
            self.set_project_state("Dance video imported", "#9ad0a0")
        except Exception as e:
            messagebox.showerror("Video Copy Error", f"Failed to import video:\n{e}")
            self.log(f"ERROR importing video: {e}")
            self.set_project_state("Video import failed", "#e8b4b4")

    def copy_prototype_to_selected_dance(self):
        try:
            selected = copy_prototype_to_selected(self.repo_root)
            self.log(f"Copied prototype video into selected dance: {selected['id']}")
            self.refresh_status(log_message=False)
            self.set_project_state("Prototype copied to selected dance", "#9ad0a0")
        except Exception as e:
            messagebox.showerror("Prototype Copy Error", f"Failed to copy prototype video:\n{e}")
            self.log(f"ERROR copying prototype video: {e}")
            self.set_project_state("Prototype copy failed", "#e8b4b4")

    def generate_reference_for_selected_dance(self):
        if self.process_running:
            messagebox.showwarning("Busy", "Another project task is already running.")
            return

        selected = get_selected_dance(self.repo_root)
        if selected is None:
            messagebox.showerror("No Dance Selected", "No dance folders were found in data/dances.")
            return

        self.process_running = True
        self.set_project_state(f"Generating reference: {selected['name']}", "#e8c98c")
        self.log(f"Preparing selected dance for extraction: {selected['id']}")

        thread = threading.Thread(target=self._generate_reference_thread, daemon=True)
        thread.start()

    def _generate_reference_thread(self):
        try:
            selected = prepare_selected_dance_for_extraction(self.repo_root)

            result = subprocess.run(
                [sys.executable, str(self.repo_root / "extract_reference.py")],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )

            if result.stdout.strip():
                self.root.after(0, lambda: self.log(result.stdout.strip()))
            if result.stderr.strip():
                self.root.after(0, lambda: self.log(result.stderr.strip()))

            if result.returncode != 0:
                self.root.after(
                    0,
                    lambda: self._handle_generation_failure(
                        f"Reference generation failed with code {result.returncode}."
                    ),
                )
                return

            stored = store_generated_reference_for_selected(self.repo_root)

            self.root.after(
                0,
                lambda: self._handle_generation_success(stored["name"]),
            )
        except Exception as e:
            self.root.after(0, lambda: self._handle_generation_failure(str(e)))

    def _handle_generation_success(self, dance_name):
        self.process_running = False
        self.log(f"Reference saved for selected dance: {dance_name}")
        self.refresh_status(log_message=False)
        self.set_project_state("Selected dance reference generated", "#9ad0a0")
        messagebox.showinfo("Reference Ready", f"Reference generated for {dance_name}.")

    def _handle_generation_failure(self, error_message):
        self.process_running = False
        self.log(f"ERROR generating selected dance reference: {error_message}")
        self.refresh_status(log_message=False)
        self.set_project_state("Reference generation failed", "#e8b4b4")
        messagebox.showerror("Reference Generation Error", error_message)

    def run_script(self, script_name, label):
        if self.process_running:
            messagebox.showwarning("Busy", "Another project task is already running.")
            return

        selected = get_selected_dance(self.repo_root)
        if selected is None:
            messagebox.showerror("No Dance Selected", "No dances were found in data/dances.")
            return

        save_selected_dance(self.repo_root, selected["id"])

        script_path = self.repo_root / script_name
        if not script_path.exists():
            messagebox.showerror("Missing Script", f"Could not find:\n{script_path}")
            self.log(f"ERROR: Missing script {script_path}")
            return

        self.process_running = True
        self.set_project_state(f"Running: {label}", "#e8c98c")
        self.log(f"Launching {label} for selected dance: {selected['name']}")

        thread = threading.Thread(
            target=self._run_script_thread,
            args=(script_path, label),
            daemon=True,
        )
        thread.start()

    def _run_script_thread(self, script_path, label):
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
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
                f"{label} finished with return code {result.returncode}.\nCheck the log panel for details.",
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
