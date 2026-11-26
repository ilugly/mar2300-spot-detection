"""Forms for the remaining MAR2300 command-line workflows."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, ttk


@dataclass(frozen=True)
class Field:
    label: str
    kind: str
    flag: str = ""
    suffix: str = ""
    default: str = ""
    choices: tuple[str, ...] = ()
    optional: bool = False


IMAGE = Field("Кадр MAR2300", "file", suffix=".mar2300")
MANIFEST = Field("Манифест разметки", "file", suffix=".jsonl")
REFERENCE = Field("Файл DRX", "file", suffix=".drx")
CHECKPOINT = Field("Файл весов модели", "file", suffix=".pt")
DEVICE = Field("Устройство", "choice", "--device", default="cpu", choices=("cpu", "cuda", "mps"))
DIRECTORY = Field("Папка одной серии", "folder")
POINTS = Field("Точки MVView (X Y, по одной на строку)", "points", "--point")
CALIBRATION = Field("Калибровка детектора (JSON, если есть)", "file", "--calibration", ".json", optional=True)
ALLOW_MISSING = Field("Пропускать записи без исходного кадра", "check", "--allow-missing")

OPERATIONS = {
    "Сведения о кадре": (
        "inspect", "Показывает размеры, диапазон отсчётов и заголовок кадра.", (IMAGE,),
    ),
    "Проверка файла DRX": (
        "inspect-drx", "Сопоставляет отражения с кадрами одной серии по углу φ.",
        (DIRECTORY, REFERENCE, CALIBRATION, ALLOW_MISSING,
         Field("Сверять интенсивность с пикселем кадра", "check", "--verify-pixels", default="1")),
    ),
    "Экспорт точек MVView": (
        "export-drx", "Введите координаты из MVView. Точка 1 1 — нижний левый пиксель.",
        (IMAGE, Field("Новый файл DRX", "save", suffix=".drx"), POINTS, CALIBRATION),
    ),
    "Калибровка детектора": (
        "calibrate-offset", "Введите точки в порядке строк в контрольном DRX, затем сохраните отчёт JSON.",
        (IMAGE, REFERENCE, POINTS),
    ),
    "Импорт разметки из DRX": (
        "import-drx", "Укажите папку с сериями. В каждой серии нужен один файл DRX.",
        (Field("Общая папка серий", "folder"),
         Field("Новый манифест разметки", "save", suffix=".jsonl"),
         Field("Обучающие серии (через запятую)", "list", "--train"),
         Field("Валидационные серии (через запятую)", "list", "--validation", optional=True),
         Field("Тестовые серии (через запятую)", "list", "--test", optional=True),
         Field("Диапазоны φ (серия, от, до, по одному на строку)", "angles", "--phi-range", optional=True),
         ALLOW_MISSING),
    ),
    "Проверка манифеста": (
        "validate", "Проверяет чтение кадров, границы точек и разбиение по экспериментам.", (MANIFEST,),
    ),
    "Обучение модели": (
        "train", "Обучает SpotCNN по манифесту и сохраняет новый файл весов.",
        (MANIFEST, Field("Новый файл весов", "save", suffix=".pt"), DEVICE,
         Field("Эпохи", "entry", "--epochs", default="12"),
         Field("Начальное число генератора (seed)", "entry", "--seed", default="17")),
    ),
    "Оценка модели": (
        "evaluate", "Сравнивает кандидатов модели с отмеченными точками выбранной выборки.",
        (MANIFEST, CHECKPOINT, DEVICE,
         Field("Выборка", "choice", "--split", default="validation", choices=("validation", "test")),
         Field("Ранжирование", "choice", "--ranking", default="contrast", choices=("contrast", "cnn")),
         Field("Мин. оценка (0–1)", "entry", "--min-score", default="0"),
         Field("Допуск совпадения, пикс.", "entry", "--match-radius-px", default="5"),
         Field("Показать детали по кадрам", "check", "--details")),
    ),
}


def run_command(arguments):
    process = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", "from mar_spots.cli import main; raise SystemExit(main())", *arguments],
        capture_output=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.returncode:
        details = (process.stderr or process.stdout).strip()
        raise ValueError(details.splitlines()[-1] if details else f"Код завершения: {process.returncode}")
    return process.stdout


class Tools:
    def __init__(self, page, task, set_status, controls):
        self.root = page.winfo_toplevel()
        self.task = task
        self.set_status = set_status
        self.controls = controls
        self.fields = []
        self.form_controls = []
        self.report = ""

        page.columnconfigure(0, weight=1, uniform="panels")
        page.columnconfigure(1, weight=1, uniform="panels")
        page.rowconfigure(0, weight=1)
        form = ttk.Frame(page, style="Card.TFrame", padding=14)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(form, text="Операция", style="Title.TLabel").pack(anchor="w")
        self.operation = ttk.Combobox(form, values=tuple(OPERATIONS), state="readonly")
        self.operation.current(0)
        self.operation.pack(fill="x", pady=(7, 6))
        self.operation.bind("<<ComboboxSelected>>", self.build_fields)
        controls.append((self.operation, "readonly"))
        self.intro = ttk.Label(form, style="Muted.TLabel", wraplength=410)
        self.intro.pack(anchor="w", pady=(0, 7))
        self.form = ttk.Frame(form)
        self.form.pack(fill="x", expand=True, anchor="n")
        run = ttk.Button(form, text="Выполнить", style="Primary.TButton", command=self.run)
        run.pack(fill="x", side="bottom", pady=(7, 0))
        controls.append((run, "normal"))

        output = ttk.Frame(page, style="Card.TFrame", padding=14)
        output.grid(row=0, column=1, sticky="nsew")
        top = ttk.Frame(output)
        top.pack(fill="x", pady=(0, 7))
        ttk.Label(top, text="Отчёт", style="Title.TLabel").pack(side="left")
        self.save = ttk.Button(top, text="Сохранить JSON…", command=self.save_report, state="disabled")
        self.save.pack(side="right")
        text_frame = ttk.Frame(output)
        text_frame.pack(fill="both", expand=True)
        self.output = tk.Text(text_frame, width=1, height=1, wrap="word", state="disabled", relief="flat",
                              highlightthickness=1, highlightbackground="#dce4ee", background="#f7f9fc",
                              foreground="#17273e", padx=9, pady=8, font=("TkFixedFont", -13))
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)
        self.build_fields()

    def build_fields(self, _event=None):
        for control in self.form_controls:
            self.controls.remove(control)
        self.form_controls.clear()
        self.fields.clear()
        for widget in self.form.winfo_children():
            widget.destroy()
        _, description, fields = OPERATIONS[self.operation.get()]
        self.intro.configure(text=description)
        self.set_report("")
        for field in fields:
            self.build_field(field)

    def build_field(self, field):
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=(0, 4))
        value = tk.StringVar(value=field.default)
        if field.kind == "check":
            widget = ttk.Checkbutton(row, text=field.label, variable=value, onvalue="1", offvalue="0")
            widget.pack(anchor="w")
        else:
            ttk.Label(row, text=field.label, style="Field.TLabel").pack(anchor="w", pady=(0, 1))
            if field.kind == "choice":
                widget = ttk.Combobox(row, textvariable=value, values=field.choices, state="readonly")
            elif field.kind in ("points", "angles"):
                widget = tk.Text(row, height=2, width=1, relief="solid", borderwidth=1, undo=True)
            else:
                widget = ttk.Entry(row, textvariable=value, width=10)
            widget.pack(side="left", fill="x", expand=True)
            if field.kind in ("file", "folder", "save"):
                button = ttk.Button(row, text="Обзор…", command=lambda: self.browse(field, value))
                button.pack(side="right", padx=(5, 0))
                self.form_controls.append((button, "normal"))

        self.form_controls.append((widget, "readonly" if field.kind == "choice" else "normal"))
        self.controls.extend(self.form_controls[-(2 if field.kind in ("file", "folder", "save") else 1):])
        self.fields.append((field, value, widget))

    def browse(self, field, value):
        if field.kind == "folder":
            path = filedialog.askdirectory(parent=self.root, mustexist=True)
        else:
            pattern = "*.pt *.pth" if field.suffix == ".pt" else f"*{field.suffix}"
            options = {"parent": self.root, "filetypes": ((pattern, pattern), ("Все файлы", "*"))}
            if field.kind == "save":
                path = filedialog.asksaveasfilename(**options, defaultextension=field.suffix, confirmoverwrite=False)
            else:
                path = filedialog.askopenfilename(**options)
        if path:
            value.set(path)

    def arguments(self):
        command, _, _ = OPERATIONS[self.operation.get()]
        positional, options = [], []
        for field, variable, widget in self.fields:
            if field.kind in ("points", "angles"):
                value = widget.get("1.0", "end").strip()
            else:
                value = variable.get().strip()
            if field.kind == "check":
                if value == "1":
                    options.append(field.flag)
                continue
            if not value:
                if not field.optional:
                    raise ValueError(f"Заполните поле «{field.label}».")
                continue
            if field.kind in ("points", "angles"):
                for line in value.splitlines():
                    if line.strip():
                        values = line.split() if field.kind == "points" else [part.strip() for part in line.split(",")]
                        expected = 2 if field.kind == "points" else 3
                        if len(values) != expected or not all(values):
                            raise ValueError(f"Проверьте строку «{line}» в поле «{field.label}».")
                        options.extend((field.flag, *values))
                continue
            if field.kind in ("file", "folder", "save"):
                path = Path(value).expanduser()
                if field.kind == "save" and path.exists():
                    raise ValueError(f"Файл уже существует. Укажите другое имя: {path}")
                value = str(path)
            if field.kind == "list":
                options.extend((field.flag, *(item.strip() for item in value.split(",") if item.strip())))
            elif field.flag:
                options.extend((field.flag, value.replace(",", ".") if field.kind == "entry" else value))
            else:
                positional.append(value)
        return [command, *positional, *options]

    def run(self):
        try:
            arguments = self.arguments()
        except ValueError as exc:
            self.set_report(str(exc), can_save=False)
            self.set_status("Проверьте поля операции.", error=True)
            return
        self.set_report("")

        def finished(report):
            self.set_report(report, can_save=True)
            self.set_status("Операция завершена. Отчёт можно сохранить в JSON.")

        self.task(lambda _post: run_command(arguments), finished, f"{self.operation.get()}…")

    def set_report(self, text, *, can_save=False):
        self.report = text if can_save else ""
        self.save.configure(state="normal" if can_save else "disabled")
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def save_report(self):
        command, _, _ = OPERATIONS[self.operation.get()]
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                            initialfile=f"{command}.json", filetypes=(("Отчёты JSON", "*.json"),))
        if path:
            try:
                Path(path).write_text(self.report, encoding="utf-8")
            except OSError as exc:
                self.set_status(f"Не удалось сохранить отчёт. {exc}", error=True)
                return
            self.set_status(f"Отчёт сохранён  •  {path}")
