import tkinter as tk
from tkinter import filedialog, messagebox


class AfroDanceAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Afro Dance Analyzer")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f4ead7")

        self.video_path = None

        self.build_header()
        self.build_main_layout()
        self.build_bottom_timeline()

    def build_header(self):
        header = tk.Frame(self.root, bg="#4b2e1e", height=70)
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="Afro Dance Analyzer",
            font=("Arial", 24, "bold"),
            bg="#4b2e1e",
            fg="#f5deb3",
            pady=15
        )
        title.pack(side="left", padx=20)

        metrics_frame = tk.Frame(header, bg="#4b2e1e")
        metrics_frame.pack(side="right", padx=20)

        tk.Label(
            metrics_frame,
            text="Accuracy Score: ______",
            font=("Arial", 12),
            bg="#4b2e1e",
            fg="white"
        ).pack(side="left", padx=10)

        tk.Label(
            metrics_frame,
            text="Rhythm Sync: ______",
            font=("Arial", 12),
            bg="#4b2e1e",
            fg="white"
        ).pack(side="left", padx=10)

    def build_main_layout(self):
        main_frame = tk.Frame(self.root, bg="#f4ead7")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel
        left_panel = tk.Frame(main_frame, bg="#2b1d17", width=280, height=450, bd=2, relief="ridge")
        left_panel.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 20))
        left_panel.grid_propagate(False)

        self.video_placeholder = tk.Label(
            left_panel,
            text="Video Preview Area\n(Early Prototype Placeholder)",
            font=("Arial", 14),
            bg="#3a2a22",
            fg="#e8d8c3",
            width=24,
            height=12,
            relief="sunken"
        )
        self.video_placeholder.pack(pady=20)

        button_frame = tk.Frame(left_panel, bg="#2b1d17")
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Start", bg="#3fa34d", fg="white", width=10, command=self.start_tracking).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Pause", bg="#d9822b", fg="white", width=10, command=self.pause_tracking).grid(row=0, column=1, padx=5)
        tk.Button(button_frame, text="Stop", bg="#b83232", fg="white", width=10, command=self.stop_tracking).grid(row=0, column=2, padx=5)

        self.status_label = tk.Label(
            left_panel,
            text="Tracking: Inactive",
            font=("Arial", 13, "bold"),
            bg="#2b1d17",
            fg="#f5deb3"
        )
        self.status_label.pack(pady=20)

        tk.Button(
            left_panel,
            text="Upload Video",
            font=("Arial", 12, "bold"),
            bg="#1f6aa5",
            fg="white",
            width=20,
            command=self.upload_video
        ).pack(pady=10)

        self.file_label = tk.Label(
            left_panel,
            text="No file selected",
            font=("Arial", 10),
            bg="#2b1d17",
            fg="#d9c7b0",
            wraplength=220
        )
        self.file_label.pack(pady=10)

        # Middle top
        movement_box = self.create_panel(main_frame, "Movement Accuracy", 0, 1)
        movement_placeholder = tk.Label(
            movement_box,
            text="Placeholder for chart / score",
            font=("Arial", 14),
            bg="#5c3b2a",
            fg="#f7e7ce"
        )
        movement_placeholder.pack(expand=True)

        # Right top
        rhythm_box = self.create_panel(main_frame, "Rhythm Synchronization", 0, 2)
        rhythm_placeholder = tk.Label(
            rhythm_box,
            text="Placeholder for rhythm graph",
            font=("Arial", 14),
            bg="#5c3b2a",
            fg="#f7e7ce"
        )
        rhythm_placeholder.pack(expand=True)

        # Middle bottom
        summary_box = self.create_panel(main_frame, "Performance Summary", 1, 1)
        summary_inner = tk.Frame(summary_box, bg="#5c3b2a")
        summary_inner.pack(expand=True, fill="both", padx=10, pady=10)

        tk.Label(summary_inner, text="Steps Matched", bg="#6e4a35", fg="white", width=18, height=5).grid(row=0, column=0, padx=10)
        tk.Label(summary_inner, text="Time in Sync", bg="#6e4a35", fg="white", width=18, height=5).grid(row=0, column=1, padx=10)
        tk.Label(summary_inner, text="Improvement Tips", bg="#6e4a35", fg="white", width=18, height=5).grid(row=0, column=2, padx=10)

        # Right bottom
        pose_box = self.create_panel(main_frame, "Pose Alignment", 1, 2)
        pose_placeholder = tk.Label(
            pose_box,
            text="Placeholder for body pose skeleton",
            font=("Arial", 14),
            bg="#5c3b2a",
            fg="#f7e7ce"
        )
        pose_placeholder.pack(expand=True)

    def build_bottom_timeline(self):
        timeline_frame = tk.Frame(self.root, bg="#4b2e1e", height=100, bd=2, relief="ridge")
        timeline_frame.pack(fill="x", padx=20, pady=(0, 20))
        timeline_frame.pack_propagate(False)

        title = tk.Label(
            timeline_frame,
            text="Feedback Timeline",
            font=("Arial", 16, "bold"),
            bg="#4b2e1e",
            fg="#f5deb3"
        )
        title.pack(anchor="w", padx=15, pady=(10, 0))

        canvas = tk.Canvas(timeline_frame, bg="#6a4532", height=40, highlightthickness=0)
        canvas.pack(fill="x", padx=15, pady=10)

        canvas.create_line(20, 20, 1100, 20, fill="white", width=2)

        for i in range(10):
            x = 50 + i * 100
            canvas.create_oval(x - 6, 14, x + 6, 26, fill="#e67e22", outline="")

    def create_panel(self, parent, title, row, column):
        panel = tk.Frame(parent, bg="#5c3b2a", width=320, height=220, bd=2, relief="ridge")
        panel.grid(row=row, column=column, padx=10, pady=10, sticky="nsew")
        panel.grid_propagate(False)

        title_label = tk.Label(
            panel,
            text=title,
            font=("Arial", 16, "bold"),
            bg="#5c3b2a",
            fg="#f5deb3"
        )
        title_label.pack(anchor="w", padx=10, pady=10)

        return panel

    def upload_video(self):
        file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )

        if file_path:
            self.video_path = file_path
            self.file_label.config(text=f"Selected:\n{file_path}")

    def start_tracking(self):
        self.status_label.config(text="Tracking: Active")
        messagebox.showinfo("Start", "Tracking started (prototype action).")

    def pause_tracking(self):
        self.status_label.config(text="Tracking: Paused")
        messagebox.showinfo("Pause", "Tracking paused (prototype action).")

    def stop_tracking(self):
        self.status_label.config(text="Tracking: Stopped")
        messagebox.showinfo("Stop", "Tracking stopped (prototype action).")


if __name__ == "__main__":
    root = tk.Tk()
    app = AfroDanceAnalyzerUI(root)
    root.mainloop()