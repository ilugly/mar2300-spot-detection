"""Desktop workflow for inspecting MAR2300 frames and exporting detections."""

import argparse
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .cli import detector_offset, scan_reflections, series_frames
from .gui_tools import Tools
from .image import read_mar2300

BACKGROUND = "#eef2f7"
PAPER = "#ffffff"
INK = "#17273e"
MUTED = "#536479"
ACCENT = "#1753bc"
BORDER = "#dce4ee"
VIEW = "#111c2c"
MARK = "#4ce0ed"
SELECTED = "#ffd26b"

RANKINGS = {"Модель + контраст": "contrast", "Только модель": "cnn"}
DEFAULT_MODELS = (
    Path("spotcnn.pt"), Path("runs/live-demo/spotcnn.pt"), Path("runs/received-baseline/spotcnn.pt"),
)


def configure_styles(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=BACKGROUND)
    font = ("TkDefaultFont", -14)
    style.configure(".", font=font, background=PAPER, foreground=INK)
    style.configure("App.TFrame", background=BACKGROUND)
    style.configure("Paper.TFrame", background=PAPER)
    style.configure("Card.TFrame", background=PAPER, borderwidth=1, relief="solid", bordercolor=BORDER)
    style.configure("Hero.TLabel", background=BACKGROUND, font=("TkDefaultFont", -29, "bold"))
    style.configure("Subtitle.TLabel", background=BACKGROUND, foreground=MUTED)
    style.configure("Title.TLabel", font=("TkDefaultFont", -18, "bold"))
    style.configure("Field.TLabel", font=("TkDefaultFont", -13, "bold"))
    style.configure("Muted.TLabel", foreground=MUTED, font=("TkDefaultFont", -13))
    style.configure("Status.TLabel", background=BACKGROUND, foreground=MUTED)
    style.configure("Error.TLabel", background=BACKGROUND, foreground="#a22332")
    style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(18, 9), background="#e1e8f2", foreground=MUTED)
    style.map("TNotebook.Tab", background=[("selected", PAPER)], foreground=[("selected", INK)])
    style.configure("TButton", padding=(10, 6), background="#edf2fa", bordercolor=BORDER,
                    lightcolor="#edf2fa", darkcolor="#edf2fa")
    style.map("TButton", background=[("pressed", "#d8e3f4"), ("active", "#e0e9f8")])
    style.configure("Primary.TButton", background=ACCENT, foreground=PAPER,
                    bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                    font=("TkDefaultFont", -14, "bold"))
    style.map("Primary.TButton",
              foreground=[("disabled", MUTED), ("!disabled", PAPER)],
              background=[("disabled", "#e3eaf5"), ("pressed", "#123f92"), ("active", "#1048a9")])
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(name, padding=5, fieldbackground=PAPER, bordercolor=BORDER, arrowsize=13)
    style.map("TCombobox", fieldbackground=[("readonly", PAPER)], foreground=[("readonly", INK)])
    style.map("TCheckbutton", background=[("disabled", PAPER)])
    style.configure("Treeview", rowheight=25, borderwidth=0, fieldbackground=PAPER)
    style.configure("Treeview.Heading", background="#edf2fa", relief="flat", font=("TkDefaultFont", -13, "bold"))
    style.map("Treeview", background=[("selected", "#dfeaff")], foreground=[("selected", INK)])
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#dce5f2", borderwidth=0)


def image_preview(path):
    counts = read_mar2300(path).counts
    values = np.log1p(counts.astype(np.float32))
    sample = values[::4, ::4]
    sample = sample[sample > 0]
    if sample.size:
        low, high = np.percentile(sample, (2, 99.8))
        values = (values - low) / max(high - low, 1.0)
    gray = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(gray)


def open_folder(path):
    if sys.platform == "win32":
        from os import startfile
        startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


class App:
    def __init__(self, root, checkpoint):
        self.root = root
        root.title("MAR2300 — Поиск пятен")
        root.minsize(950, 690)
        width = min(1240, root.winfo_screenwidth() - 60)
        height = min(820, root.winfo_screenheight() - 85)
        x = max(0, (root.winfo_screenwidth() - width) // 2)
        root.geometry(f"{width}x{height}+{x}+36")
        root.protocol("WM_DELETE_WINDOW", self.close)
        configure_styles(root)

        self.source = None
        self.frames = []
        self.candidates = {}
        self.drx = None
        self.saved_to = None
        self.preview = None
        self.photo = None
        self.markers = {}
        self.selection = None
        self.busy = False
        self.controls = []

        self.frame_name = tk.StringVar()
        self.frame_info = tk.StringVar(value="Выберите кадр или папку серии")
        self.result_info = tk.StringVar(value="Кандидаты появятся после поиска")
        self.source_name = tk.StringVar(value="Источник не выбран")
        self.model_name = tk.StringVar(value=str(checkpoint) if checkpoint else "")
        self.limit = tk.StringVar(value="25")
        self.min_score = tk.StringVar(value="0")
        self.ranking = tk.StringVar(value=next(iter(RANKINGS)))
        self.device = tk.StringVar(value="cpu")
        self.calibration = tk.StringVar()
        self.zoom = tk.BooleanVar()
        self.status = tk.StringVar(value="Выберите кадр .mar2300 или папку с одной серией.")

        app = ttk.Frame(root, style="App.TFrame", padding=(22, 15, 22, 12))
        app.pack(fill="both", expand=True)
        ttk.Label(app, text="MAR2300", style="Hero.TLabel").pack(anchor="w")
        ttk.Label(app, text="Поиск дифракционных пятен и выгрузка отражений",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(2, 14))

        footer = ttk.Frame(app, style="App.TFrame")
        footer.pack(side="bottom", fill="x", pady=(7, 0))
        self.status_label = ttk.Label(footer, textvariable=self.status, style="Status.TLabel", wraplength=750)
        self.status_label.pack(side="left")
        self.progress = ttk.Progressbar(footer, length=152)
        self.progress.pack(side="right")

        pages = ttk.Notebook(app)
        pages.pack(fill="both", expand=True)
        pages.enable_traversal()
        detect_page = ttk.Frame(pages, padding=12, style="App.TFrame")
        tools_page = ttk.Frame(pages, padding=12, style="App.TFrame")
        pages.add(detect_page, text="Поиск пятен")
        pages.add(tools_page, text="Инструменты")
        self.build_detection(detect_page)
        self.tools = Tools(tools_page, self.task, self.set_status, self.controls)

        def run_detection(_event):
            if pages.select() == str(detect_page):
                self.detect_button.invoke()

        root.bind("<Control-Return>", run_detection)

    def build_detection(self, page):
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, minsize=362)
        page.rowconfigure(0, weight=1)

        viewer = ttk.Frame(page, style="Card.TFrame", padding=14)
        viewer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_bar = ttk.Frame(viewer)
        preview_bar.pack(fill="x")
        ttk.Label(preview_bar, text="Кадр детектора", style="Title.TLabel").pack(side="left")
        self.crop = ttk.Checkbutton(preview_bar, text="Фрагмент", variable=self.zoom, command=self.draw)
        self.crop.pack(side="right")
        self.crop.state(["disabled"])
        self.preview_choice = ttk.Combobox(viewer, textvariable=self.frame_name, state="disabled", width=35)
        self.preview_choice.pack(fill="x", pady=(9, 7))
        self.preview_choice.bind("<<ComboboxSelected>>", self.select_frame)

        self.canvas = tk.Canvas(viewer, background=VIEW, highlightthickness=0, width=1, height=1)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.draw)
        self.canvas.bind("<Button-1>", self.select_marker)

        ttk.Label(viewer, textvariable=self.frame_info, style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(viewer, text="Бирюзовые метки — кандидаты. Координаты отсчитываются от верхнего левого пикселя (0, 0).",
                  style="Muted.TLabel", wraplength=485).pack(anchor="w", pady=(2, 0))

        side = ttk.Frame(page, style="App.TFrame")
        side.grid(row=0, column=1, sticky="nsew")
        form = ttk.Frame(side, style="Card.TFrame", padding=12)
        form.pack(fill="x")
        ttk.Label(form, text="Настройки поиска", style="Title.TLabel").pack(anchor="w", pady=(0, 7))
        ttk.Label(form, textvariable=self.source_name, wraplength=332).pack(anchor="w")
        source_buttons = ttk.Frame(form)
        source_buttons.pack(fill="x", pady=(5, 8))
        for title, command in (("Открыть кадр…", self.choose_image), ("Открыть папку…", self.choose_series)):
            button = ttk.Button(source_buttons, text=title, command=command)
            button.pack(side="left", padx=(0, 6))
            self.controls.append((button, "normal"))

        ttk.Label(form, text="Файл весов модели", style="Field.TLabel").pack(anchor="w")
        self.path_entry(form, self.model_name, (("Веса PyTorch", "*.pt *.pth"), ("Все файлы", "*")))
        settings = ttk.Frame(form)
        settings.pack(fill="x")
        for column in range(2):
            settings.columnconfigure(column, weight=1, uniform="parameters")
        self.parameter(settings, 0, 0, "Кандидатов на кадр", self.limit)
        self.parameter(settings, 0, 1, "Мин. оценка (0–1)", self.min_score)
        self.parameter(settings, 2, 0, "Ранжирование", self.ranking, tuple(RANKINGS))
        self.parameter(settings, 2, 1, "Устройство", self.device, ("cpu", "cuda", "mps"))

        ttk.Label(form, text="Калибровка детектора (JSON, если есть)", style="Field.TLabel").pack(anchor="w", pady=(4, 0))
        self.path_entry(form, self.calibration, (("Калибровка", "*.json"), ("Все файлы", "*")))

        self.detect_button = ttk.Button(form, text="Найти пятна", style="Primary.TButton", command=self.detect)
        self.detect_button.pack(fill="x", pady=(3, 0))

        results = ttk.Frame(side, style="Card.TFrame", padding=12)
        results.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(results, text="Результаты", style="Title.TLabel").pack(anchor="w")
        ttk.Label(results, textvariable=self.result_info, style="Muted.TLabel", wraplength=334).pack(anchor="w", pady=(2, 6))
        actions = ttk.Frame(results)
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        self.save_button = ttk.Button(actions, text="Сохранить DRX…", command=self.save)
        self.save_button.pack(side="left")
        self.folder_button = ttk.Button(actions, text="Папка результата", command=self.open_result)
        self.folder_button.pack(side="right")

        table = ttk.Frame(results)
        table.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table, columns=("x", "y", "score"), show="headings",
                                  selectmode="browse", height=3)
        for column, heading, width in (("x", "X", 61), ("y", "Y", 61), ("score", "Оценка", 84)):
            self.table.heading(column, text=heading)
            self.table.column(column, width=width, minwidth=40, anchor="center")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self.select_candidate)
        self.table.bind("<Double-1>", self.show_crop)

        self.update_actions()

    def parameter(self, parent, row, column, title, value, choices=()):
        ttk.Label(parent, text=title, style="Field.TLabel").grid(
            row=row, column=column, sticky="w", padx=(0, 8), pady=(0, 2))
        if choices:
            widget = ttk.Combobox(parent, textvariable=value, values=choices, state="readonly", width=16)
        else:
            widget = ttk.Entry(parent, textvariable=value, width=16)
        widget.grid(row=row + 1, column=column, sticky="ew", padx=(0, 8), pady=(0, 6))
        self.controls.append((widget, "readonly" if choices else "normal"))

    def path_entry(self, parent, value, filetypes):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(3, 8))
        entry = ttk.Entry(row, textvariable=value, width=24)
        entry.pack(side="left", fill="x", expand=True)
        self.root.after_idle(lambda: entry.xview_moveto(1))

        def browse():
            path = filedialog.askopenfilename(parent=self.root, filetypes=filetypes)
            if path:
                value.set(path)
                entry.xview_moveto(1)

        button = ttk.Button(row, text="Обзор…", command=browse)
        button.pack(side="right", padx=(6, 0))
        self.controls.extend(((entry, "normal"), (button, "normal")))

    def choose_image(self):
        path = filedialog.askopenfilename(parent=self.root,
                                          filetypes=(("Кадры MAR2300", "*.mar2300 *.MAR2300"), ("Все файлы", "*")))
        if path:
            self.set_source(Path(path))

    def choose_series(self):
        path = filedialog.askdirectory(parent=self.root, mustexist=True)
        if path:
            self.set_source(Path(path))

    def set_source(self, path):
        try:
            frames = series_frames(path) if path.is_dir() else [path] if path.is_file() else []
            if not frames:
                raise ValueError("В выбранном источнике нет кадров .mar2300.")
            if frames[0].suffix.lower() != ".mar2300":
                raise ValueError("Выберите файл с расширением .mar2300.")
        except (OSError, ValueError) as exc:
            self.error("Не удалось открыть источник", exc)
            return
        self.source = path
        self.frames = frames
        suffix = f"  •  кадров в серии: {len(frames)}" if path.is_dir() else ""
        self.source_name.set(path.name + suffix)
        self.drx = self.saved_to = None
        self.candidates = {}
        self.preview_choice.configure(values=[frame.name for frame in frames])
        self.preview_choice.current(0)
        self.result_info.set("Кандидаты появятся после поиска")
        self.select_frame()

    def select_frame(self, _event=None):
        index = self.preview_choice.current()
        if index < 0:
            return
        path = self.frames[index]
        self.preview = None
        self.zoom.set(False)
        self.update_table()
        self.frame_info.set(f"Чтение {path.name}…")
        self.draw()

        def loaded(preview):
            self.preview = preview
            self.frame_info.set(f"{path.name}  •  {preview.width} × {preview.height} пикс.")
            self.set_status("Кадр открыт." if self.drx else "Кадр открыт. Выберите веса модели и нажмите «Найти пятна».")
            self.draw()

        def failed():
            self.frame_info.set(f"Не удалось открыть {path.name}")
            self.draw()

        self.task(lambda _post: image_preview(path), loaded, "Чтение кадра…", failed)

    def update_table(self):
        self.selection = None
        self.crop.state(["disabled"])
        self.table.delete(*self.table.get_children())
        for index, point in enumerate(self.visible_points()):
            self.table.insert("", "end", iid=str(index), values=(point["x"], point["y"], f'{point["score"]:.4f}'))

    def visible_points(self):
        index = self.preview_choice.current()
        return self.candidates.get(self.frames[index], []) if 0 <= index < len(self.frames) else []

    def select_candidate(self, _event=None):
        selected = self.table.selection()
        if selected:
            self.selection = int(selected[0])
            self.crop.state(["!disabled"])
            self.draw()

    def select_marker(self, event):
        if self.markers:
            def distance(index):
                x, y = self.markers[index]
                return (x - event.x) ** 2 + (y - event.y) ** 2

            index = min(self.markers, key=distance)
            if distance(index) <= 100:
                self.table.selection_set(str(index))
                self.table.see(str(index))
                self.table.focus(str(index))

    def show_crop(self, _event=None):
        if self.selection is not None:
            self.zoom.set(True)
            self.draw()

    def draw(self, _event=None):
        canvas = self.canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("all")
        self.markers.clear()
        if self.preview is None:
            canvas.create_text(width / 2, height / 2, text="Предпросмотр кадра", fill="#a9b8cd",
                               font=("TkDefaultFont", -16))
            return

        points = self.visible_points()
        zoom = self.zoom.get() and self.selection is not None
        left, top = 0, 0
        preview = self.preview
        if zoom:
            center = points[self.selection]
            left = max(0, min(center["x"] - 32, preview.width - 65))
            top = max(0, min(center["y"] - 32, preview.height - 65))
            preview = preview.crop((left, top, min(left + 65, preview.width), min(top + 65, preview.height)))

        scale = min(max(1, width - 20) / preview.width, max(1, height - 20) / preview.height)
        target = (max(1, round(preview.width * scale)), max(1, round(preview.height * scale)))
        resized = preview.resize(target, Image.Resampling.NEAREST if zoom else Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(resized, master=self.root)
        x0, y0 = (width - target[0]) / 2, (height - target[1]) / 2
        canvas.create_image(x0, y0, image=self.photo, anchor="nw")
        for index, point in enumerate(points):
            local_x, local_y = point["x"] - left, point["y"] - top
            if not (0 <= local_x < preview.width and 0 <= local_y < preview.height):
                continue
            x = x0 + (local_x + 0.5) * target[0] / preview.width
            y = y0 + (local_y + 0.5) * target[1] / preview.height
            self.markers[index] = x, y
            selected = index == self.selection
            radius = 7 if selected else 4
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                               outline=SELECTED if selected else MARK, width=2 if selected else 1.5)
        if zoom:
            note = f"Фрагмент  •  X {left}–{left + preview.width - 1}  ·  Y {top}–{top + preview.height - 1}"
            canvas.create_rectangle(0, height - 26, width, height, fill=VIEW, outline="")
            canvas.create_text(12, height - 13, text=note, fill=PAPER, anchor="w")

    def detect(self):
        source = self.source
        try:
            limit = int(self.limit.get())
            min_score = float(self.min_score.get().replace(",", "."))
            if limit < 1 or not 0 <= min_score <= 1:
                raise ValueError
        except ValueError:
            self.error("Проверьте параметры поиска", "Лимит должен быть целым числом от 1, оценка — числом от 0 до 1.")
            return
        model_name = self.model_name.get().strip()
        checkpoint = Path(model_name).expanduser()
        if not model_name or not checkpoint.is_file():
            self.error("Нужен файл весов", "Укажите существующий файл весов SpotCNN, например spotcnn.pt.")
            return
        calibration_name = self.calibration.get().strip()
        calibration = Path(calibration_name).expanduser() if calibration_name else None
        ranking = RANKINGS[self.ranking.get()]
        device = self.device.get()
        total = len(self.frames)

        def work(post):
            try:
                from . import learning
            except ModuleNotFoundError as exc:
                if exc.name != "torch":
                    raise
                raise ValueError("Установите зависимости модели командой python -m pip install '.[ml]'.") from exc
            if ((device == "cuda" and not learning.torch.cuda.is_available())
                    or (device == "mps" and not learning.torch.backends.mps.is_available())):
                raise ValueError(f"Устройство {device} недоступно. Выберите cpu.")
            offset = detector_offset(argparse.Namespace(calibration=calibration, offset_mm=(0, 0)))
            model = learning.load_model(checkpoint, device)
            candidates = {}

            def on_frame(path, points):
                candidates[path] = points
                done = len(candidates)
                post(lambda: self.show_progress(done, total, path.name))

            drx, frames = scan_reflections(
                source, offset,
                lambda counts: learning.predict(model, counts, limit, min_score=min_score, ranking=ranking),
                on_frame=on_frame,
            )
            return drx, frames, candidates

        def finished(result):
            self.drx, frames, self.candidates = result
            self.saved_to = None
            self.zoom.set(False)
            self.update_table()
            self.draw()
            count = len(self.drx.reflections)
            self.result_info.set(f"Обработано кадров: {len(frames)}  •  Кандидатов: {count}")
            self.set_status("Поиск завершён. Сохраните результат в DRX." if count else
                            "Кандидатов нет. DRX будет содержать только длину волны.")

        self.task(work, finished, "Загрузка модели…")

    def show_progress(self, done, total, name):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=done, maximum=total)
        self.set_status(f"Обработано кадров: {done} из {total}  •  {name}")

    def save(self):
        if self.drx is None:
            return
        name = self.source.name if self.source.is_dir() else self.source.stem
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".drx",
                                            filetypes=(("Файлы DRX", "*.drx"),),
                                            initialfile=f"{name}-detected.drx", confirmoverwrite=False)
        if not path:
            return
        try:
            self.drx.write(Path(path))
        except FileExistsError:
            self.error("Файл уже существует", "Укажите другое имя для нового результата.")
            return
        except OSError as exc:
            self.error("Не удалось сохранить DRX", exc)
            return
        self.saved_to = Path(path)
        self.update_actions()
        self.set_status(f"DRX сохранён  •  {path}")

    def open_result(self):
        try:
            open_folder(self.saved_to.parent)
        except OSError as exc:
            self.error("Не удалось открыть папку", exc)

    def set_status(self, text, error=False):
        self.status.set(text)
        self.status_label.configure(style="Error.TLabel" if error else "Status.TLabel")

    def error(self, title, exc):
        details = str(exc).strip() or type(exc).__name__
        self.set_status(title, error=True)
        messagebox.showerror(title, details, parent=self.root)

    def update_actions(self):
        self.preview_choice.configure(state="readonly" if self.frames and not self.busy else "disabled")
        for button, ready in ((self.detect_button, self.source),
                              (self.save_button, self.drx), (self.folder_button, self.saved_to)):
            button.configure(state="normal" if ready is not None and not self.busy else "disabled")

    def task(self, work, success, status, failure=None):
        if self.busy:
            return
        self.busy = True
        self.update_actions()
        for widget, _state in self.controls:
            widget.configure(state="disabled")
        self.set_status(status)
        self.progress.configure(mode="indeterminate")
        self.progress.start(13)
        messages = Queue()

        def run():
            try:
                result = work(lambda action: messages.put(("progress", action)))
            except Exception as exc:
                messages.put(("error", exc))
            else:
                messages.put(("done", result))

        def poll():
            while True:
                try:
                    kind, payload = messages.get_nowait()
                except Empty:
                    self.root.after(45, poll)
                    return
                if kind == "progress":
                    payload()
                    continue
                self.progress.stop()
                self.progress.configure(value=0)
                self.busy = False
                for widget, state in self.controls:
                    widget.configure(state=state)
                if kind == "error":
                    if failure is not None:
                        failure()
                    self.error("Операция не выполнена", payload)
                else:
                    success(payload)
                self.update_actions()
                return

        Thread(target=run, daemon=True).start()
        self.root.after(45, poll)

    def close(self):
        if self.busy:
            messagebox.showinfo("Операция выполняется", "Дождитесь завершения текущей операции.", parent=self.root)
        else:
            self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Графический интерфейс MAR2300")
    parser.add_argument("source", nargs="?", type=Path, help="Кадр или папка серии")
    parser.add_argument("--checkpoint", type=Path, help="Файл весов SpotCNN")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or next((path.resolve() for path in DEFAULT_MODELS if path.is_file()), None)
    root = tk.Tk()
    app = App(root, checkpoint)
    if args.source:
        root.after_idle(lambda: app.set_source(args.source.expanduser()))
    root.mainloop()


if __name__ == "__main__":
    main()
