"""Minimal Tkinter front-end. It only builds a command line and runs `python -m badapple_dr run ...`
in a subprocess, streaming its log; the command-line tool does all the work (and can resume)."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    root = tk.Tk()
    root.title("Bad Apple!! x DELTARUNE - frame matcher")
    root.geometry("980x700")
    vars_ = {k: tk.StringVar() for k in ("reference", "source", "export_source", "work", "out")}
    vars_["work"].set(str(Path.cwd() / "work"))
    vars_["out"].set(str(Path.cwd() / "output"))
    opts = {
        "keep_threshold": tk.StringVar(value="0.010"),
        "refine_rounds": tk.StringVar(value="2"),
        "semantic": tk.StringVar(value="auto"),
        "hwaccel": tk.StringVar(value="none"),
        "workers": tk.StringVar(value="4"),
        "ref_start": tk.StringVar(value="0"),
    }
    frm = ttk.Frame(root, padding=8)
    frm.pack(fill="both", expand=True)

    def picker(row, label, key, kind):
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=vars_[key], width=90).grid(row=row, column=1, sticky="we", padx=4)

        def choose():
            if kind == "dir":
                p = filedialog.askdirectory()
            else:
                p = filedialog.askopenfilename(filetypes=[("Video", "*.mkv *.mp4 *.webm *.mov *.avi"), ("All", "*.*")])
            if p:
                vars_[key].set(p)

        ttk.Button(frm, text="Browse...", command=choose).grid(row=row, column=2)

    picker(0, "Bad Apple!! reference video", "reference", "file")
    picker(1, "DELTARUNE walkthrough (analysis copy)", "source", "file")
    picker(2, "Optional HQ copy of the walkthrough", "export_source", "file")
    picker(3, "Work folder (caches, resumable)", "work", "dir")
    picker(4, "Output folder", "out", "dir")
    box = ttk.LabelFrame(frm, text="Settings", padding=6)
    box.grid(row=5, column=0, columnspan=3, sticky="we", pady=6)
    labels = [("keep_threshold", "Sampling threshold (lower = denser)"), ("refine_rounds", "Refinement rounds"),
              ("workers", "Parallel ffmpeg workers"), ("ref_start", "Reference start (s)")]
    for i, (k, lab) in enumerate(labels):
        ttk.Label(box, text=lab).grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=4)
        ttk.Entry(box, textvariable=opts[k], width=10).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w")
    ttk.Label(box, text="Semantic model").grid(row=2, column=0, sticky="w", padx=4)
    ttk.Combobox(box, textvariable=opts["semantic"], values=["auto", "none", "on"], width=8, state="readonly").grid(
        row=2, column=1, sticky="w")
    ttk.Label(box, text="Hardware decoding").grid(row=2, column=2, sticky="w", padx=4)
    ttk.Combobox(box, textvariable=opts["hwaccel"], values=["none", "auto", "d3d11va", "dxva2", "cuda", "qsv"],
                 width=8).grid(row=2, column=3, sticky="w")
    log = scrolledtext.ScrolledText(frm, height=22, font=("Consolas", 9))
    log.grid(row=7, column=0, columnspan=3, sticky="nsew")
    frm.rowconfigure(7, weight=1)
    frm.columnconfigure(1, weight=1)
    status = tk.StringVar(value="Idle")
    ttk.Label(frm, textvariable=status).grid(row=8, column=0, columnspan=3, sticky="w")
    q: "queue.Queue[str | None]" = queue.Queue()
    state = {"proc": None}

    def build_cmd():
        if not vars_["reference"].get() or not vars_["source"].get():
            raise ValueError("Select the reference and the source video first.")
        cmd = [sys.executable, "-u", "-m", "badapple_dr", "run", "--reference", vars_["reference"].get(),
               "--source", vars_["source"].get(), "--work", vars_["work"].get(), "--out", vars_["out"].get(),
               "--keep-threshold", opts["keep_threshold"].get(), "--refine-rounds", opts["refine_rounds"].get(),
               "--semantic", opts["semantic"].get(), "--hwaccel", opts["hwaccel"].get(),
               "--workers", opts["workers"].get(), "--ref-start", opts["ref_start"].get()]
        if vars_["export_source"].get():
            cmd += ["--export-source", vars_["export_source"].get()]
        return cmd

    def reader(proc):
        for line in proc.stdout:
            q.put(line)
        proc.wait()
        q.put(None)

    def start():
        if state["proc"] is not None:
            return
        try:
            cmd = build_cmd()
        except ValueError as e:
            messagebox.showerror("Missing input", str(e))
            return
        log.insert("end", "> " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        env = dict(os.environ, BADAPPLE_NO_TQDM="1", PYTHONIOENCODING="utf-8")
        flags = 0x08000000 if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=env, creationflags=flags, cwd=str(Path(__file__).resolve().parent.parent))
        state["proc"] = proc
        status.set("Running... (you can stop at any time; the next start resumes)")
        threading.Thread(target=reader, args=(proc,), daemon=True).start()

    def stop():
        p = state["proc"]
        if p is not None and p.poll() is None:
            # graceful: the pipeline sees work/STOP, saves its progress and exits
            try:
                Path(vars_["work"].get(), "STOP").write_text("stop")
            except OSError:
                p.terminate()
            status.set("Stopping after saving progress... (start again later to resume)")

            def force():
                if p.poll() is None:
                    p.terminate()
            root.after(60000, force)

    def poll():
        try:
            while True:
                line = q.get_nowait()
                if line is None:
                    p = state["proc"]
                    rc = p.returncode if p else None
                    status.set("Finished successfully." if rc == 0 else f"Stopped (exit code {rc}).")
                    state["proc"] = None
                    break
                log.insert("end", line)
                log.see("end")
        except queue.Empty:
            pass
        root.after(200, poll)

    def open_out():
        p = vars_["out"].get()
        if p and Path(p).exists():
            if os.name == "nt":
                os.startfile(p)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", p])

    btns = ttk.Frame(frm)
    btns.grid(row=6, column=0, columnspan=3, sticky="w")
    ttk.Button(btns, text="Start / Resume", command=start).pack(side="left", padx=2)
    ttk.Button(btns, text="Stop", command=stop).pack(side="left", padx=2)
    ttk.Button(btns, text="Open output folder", command=open_out).pack(side="left", padx=2)
    root.after(200, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
