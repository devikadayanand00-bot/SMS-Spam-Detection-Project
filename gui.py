
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from queue import Queue, Empty
from time import monotonic

from main import load_dataset, train_model


# ============================================================
# CONFIGURATION
# ============================================================

BG = "#0A1020"
PANEL = "#121C30"
CARD = "#18253B"
INPUT_BG = "#0D1729"
BORDER = "#293951"
WHITE = "#F1F5FF"
MUTED = "#92A3BD"
TEAL = "#38E0C2"
TEAL_DARK = "#1CA995"
RED = "#FF647C"
GREEN = "#48DDA8"
AMBER = "#FFC66D"
BLUE = "#7CA8FF"

FONT = "Segoe UI"


# ============================================================
# APPLICATION
# ============================================================

root = tk.Tk()
root.title("SMS Shield | Intelligent Spam Detection")
root.geometry("940x790")
root.minsize(700, 680)
root.configure(bg=BG)

model = None
training_queue = Queue()
training_started = monotonic()

animation_phase = 0
scan_phase = 0
result_after_id = None
is_scanning = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_label(parent, text, size=10, color=WHITE, bold=False, **kwargs):
    return tk.Label(
        parent,
        text=text,
        font=(FONT, size, "bold" if bold else "normal"),
        bg=parent.cget("bg"),
        fg=color,
        **kwargs,
    )


def make_button(parent, text, command, primary=False):
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=(FONT, 10, "bold"),
        bg=TEAL if primary else CARD,
        fg=BG if primary else WHITE,
        activebackground="#70F1DA" if primary else "#263852",
        activeforeground=BG if primary else WHITE,
        relief="flat",
        bd=0,
        cursor="hand2",
        padx=17,
        pady=11,
        highlightthickness=0,
    )


def set_status(text, color=MUTED):
    status_label.config(text=text, fg=color)


def get_message():
    return message_box.get("1.0", "end-1c").strip()


def update_counter(event=None):
    count = len(message_box.get("1.0", "end-1c"))
    counter_label.config(text=f"{count:,} characters")


def clear_message():
    global result_after_id, is_scanning

    if result_after_id is not None:
        root.after_cancel(result_after_id)
        result_after_id = None

    is_scanning = False
    message_box.delete("1.0", "end")
    update_counter()
    result_panel.pack_forget()
    scanning_panel.pack_forget()
    set_status("Ready when you are.", MUTED)
    message_box.focus_set()


def insert_example(message):
    message_box.delete("1.0", "end")
    message_box.insert("1.0", message)
    update_counter()
    message_box.focus_set()


# ============================================================
# TRAINING: USE THE SAME MODEL AS main.py
# ============================================================

def train_in_background():
    try:
        messages, labels = load_dataset()
        trained_model = train_model(messages, labels)
        training_queue.put(("success", trained_model))
    except Exception as exc:
        training_queue.put(("error", str(exc)))


def check_training_status():
    global model

    try:
        status, result = training_queue.get_nowait()

        if status == "success":
            model = result

            check_button.config(
                state="normal",
                text="ANALYZE MESSAGE  →",
            )
            set_status("Model trained successfully • Ready to analyze", GREEN)

            training_title.config(text="MODEL READY", fg=GREEN)
            training_detail.config(
                text="Your trained machine learning model is ready."
            )
            training_canvas.delete("all")
            training_canvas.create_rectangle(
                0, 0, 320, 6, fill=GREEN, outline=""
            )

        else:
            check_button.config(state="disabled")
            set_status("Training failed. Check the dataset or terminal.", RED)
            training_title.config(text="TRAINING FAILED", fg=RED)
            training_detail.config(text="Unable to prepare the classifier.")
            messagebox.showerror(
                "Model Training Error",
                "The model could not be trained.\n\n"
                + result
                + "\n\nCheck the terminal and your dataset.",
            )

    except Empty:
        root.after(200, check_training_status)


# ============================================================
# ANIMATIONS
# ============================================================

def animate_header():
    global animation_phase
    animation_phase += 1

    colors = [TEAL, "#43D8CF", "#55CFCB", "#43D8CF"]
    brand_label.config(fg=colors[animation_phase % len(colors)])

    root.after(450, animate_header)


def animate_training():
    global scan_phase
    scan_phase = (scan_phase + 12) % 340

    training_canvas.delete("all")
    training_canvas.create_rectangle(
        0, 0, 320, 6, fill=CARD, outline=""
    )
    training_canvas.create_rectangle(
        scan_phase - 50,
        0,
        scan_phase,
        6,
        fill=TEAL,
        outline="",
    )

    if model is None:
        root.after(70, animate_training)


def show_scanning():
    global is_scanning, scan_phase

    if not is_scanning:
        return

    scan_phase = (scan_phase + 8) % 320
    scanning_canvas.delete("all")
    scanning_canvas.create_rectangle(
        0, 0, 320, 4, fill=CARD, outline=""
    )
    scanning_canvas.create_rectangle(
        scan_phase - 45, 0, scan_phase, 4,
        fill=TEAL, outline=""
    )

    root.after(35, show_scanning)


def reveal_result():
    global result_after_id, is_scanning

    is_scanning = False
    scanning_panel.pack_forget()
    result_panel.pack(fill="x", pady=(18, 0))
    result_panel.update_idletasks()

    # Simple reveal animation: gradually brighten the result border.
    colors = [BORDER, "#31516B", "#287A79", TEAL]

    def step(index=0):
        result_panel.config(
            highlightbackground=colors[index],
            highlightcolor=colors[index],
        )
        if index + 1 < len(colors):
            root.after(65, lambda: step(index + 1))

    step()
    result_after_id = None


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_message():
    global is_scanning, result_after_id

    if model is None:
        messagebox.showinfo(
            "Model is training",
            "Please wait until the model has finished training.",
        )
        return

    message = get_message()

    if not message:
        messagebox.showwarning(
            "Empty message",
            "Enter or paste an SMS message before analyzing.",
        )
        message_box.focus_set()
        return

    if result_after_id is not None:
        root.after_cancel(result_after_id)
        result_after_id = None

    is_scanning = True
    result_panel.pack_forget()
    scanning_panel.pack(fill="x", pady=(18, 0))
    set_status("Analyzing message patterns...", TEAL)
    check_button.config(state="disabled", text="ANALYZING...")

    show_scanning()

    # Keep the interface responsive while prediction runs.
    Thread(
        target=predict_in_background,
        args=(message,),
        daemon=True,
    ).start()


def predict_in_background(message):
    try:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.named_steps["classifier"].classes_)
        spam_probability = float(
            probabilities[classes.index("spam")] * 100
        )

        root.after(
            0,
            lambda: display_result(prediction, spam_probability),
        )

    except Exception as exc:
        root.after(0, lambda: prediction_error(str(exc)))


def prediction_error(error):
    global is_scanning

    is_scanning = False
    scanning_panel.pack_forget()
    check_button.config(state="normal", text="ANALYZE MESSAGE  →")
    set_status("Analysis failed.", RED)
    messagebox.showerror("Prediction Error", error)


def display_result(prediction, spam_probability):
    global is_scanning

    is_scanning = False
    check_button.config(state="normal", text="ANALYZE MESSAGE  →")

    if prediction == "spam":
        result_icon.config(text="!", fg=RED)
        result_title.config(text="SPAM DETECTED", fg=RED)
        result_description.config(
            text="This message matches patterns learned from spam examples.",
            fg=MUTED,
        )
        result_badge.config(text="HIGH RISK CLASSIFICATION", fg=RED)
        meter_color = RED
        set_status("Analysis complete • Be cautious with this message", RED)

    else:
        result_icon.config(text="✓", fg=GREEN)
        result_title.config(text="LIKELY LEGITIMATE", fg=GREEN)
        result_description.config(
            text="The model classified this message as ham.",
            fg=MUTED,
        )
        result_badge.config(text="LIKELY LEGITIMATE CLASSIFICATION", fg=GREEN)
        meter_color = GREEN
        set_status("Analysis complete • Classification is an estimate", GREEN)

    probability_label.config(
        text=f"{spam_probability:.1f}%"
    )

    probability_bar.delete("all")
    probability_bar.update_idletasks()

    width = max(probability_bar.winfo_width(), 320)
    probability_bar.create_rectangle(
        0, 0, width, 10, fill=CARD, outline=""
    )
    probability_bar.create_rectangle(
        0, 0, width * spam_probability / 100, 10,
        fill=meter_color, outline=""
    )

    reveal_result()


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(root, bg=BG)
header.pack(fill="x", padx=34, pady=(25, 16))

brand_row = tk.Frame(header, bg=BG)
brand_row.pack(fill="x")

brand_label = make_label(
    brand_row, "◈  SMS SHIELD", 25, TEAL, True
)
brand_label.pack(side="left")

make_label(
    brand_row, "MACHINE LEARNING", 9, BLUE, True
).pack(side="right", pady=(9, 0))

make_label(
    header,
    "INTELLIGENT TEXT ANALYSIS  /  SPAM PROTECTION",
    9, MUTED, True,
).pack(anchor="w", pady=(6, 0))


# ============================================================
# MODEL STATUS CARD
# ============================================================

training_card = tk.Frame(
    root, bg=PANEL, padx=19, pady=14,
    highlightbackground=BORDER, highlightthickness=1,
)
training_card.pack(fill="x", padx=30, pady=(0, 17))

training_top = tk.Frame(training_card, bg=PANEL)
training_top.pack(fill="x")

training_title = make_label(
    training_top, "PREPARING MODEL", 10, TEAL, True
)
training_title.pack(side="left")

make_label(
    training_top, "WORD + CHARACTER TEXT FEATURES", 8, MUTED
).pack(side="right")

training_detail = make_label(
    training_card,
    "Loading dataset and training the classifier in the background...",
    9, MUTED,
)
training_detail.pack(anchor="w", pady=(7, 10))

training_canvas = tk.Canvas(
    training_card, width=320, height=6,
    bg=CARD, highlightthickness=0,
)
training_canvas.pack(fill="x")


# ============================================================
# MAIN WORKSPACE
# ============================================================

workspace = tk.Frame(
    root, bg=PANEL, padx=23, pady=21,
    highlightbackground=BORDER, highlightthickness=1,
)
workspace.pack(fill="both", expand=True, padx=30, pady=(0, 17))

title_row = tk.Frame(workspace, bg=PANEL)
title_row.pack(fill="x")

make_label(
    title_row, "Analyze a message", 18, WHITE, True
).pack(side="left")

make_label(
    title_row, "LIVE ANALYSIS", 8, TEAL, True
).pack(side="right", pady=(7, 0))

make_label(
    workspace,
    "Paste an SMS below. The trained model will estimate its spam likelihood.",
    10, MUTED,
).pack(anchor="w", pady=(7, 15))


# ============================================================
# TEXT INPUT
# ============================================================

input_card = tk.Frame(
    workspace, bg=INPUT_BG,
    highlightbackground=BORDER, highlightthickness=1,
)
input_card.pack(fill="x")

message_box = tk.Text(
    input_card,
    height=5,
    wrap="word",
    font=(FONT, 11),
    bg=INPUT_BG,
    fg=WHITE,
    insertbackground=TEAL,
    selectbackground=TEAL_DARK,
    selectforeground=WHITE,
    relief="flat",
    bd=0,
    padx=15,
    pady=14,
    undo=True,
    highlightthickness=0,
)
message_box.pack(fill="both", expand=True)
message_box.bind("<KeyRelease>", update_counter)

input_footer = tk.Frame(workspace, bg=PANEL)
input_footer.pack(fill="x", pady=(6, 15))

make_label(
    input_footer, "TEXT INPUT", 8, MUTED, True
).pack(side="left")

counter_label = make_label(
    input_footer, "0 characters", 9, MUTED
)
counter_label.pack(side="right")


# ============================================================
# ACTION BUTTONS
# ============================================================

actions = tk.Frame(workspace, bg=PANEL)
actions.pack(fill="x")

check_button = make_button(
    actions, "LOADING MODEL...", classify_message, primary=True
)
check_button.config(state="disabled")
check_button.pack(side="left")

clear_button = make_button(
    actions, "CLEAR TEXT", clear_message
)
clear_button.pack(side="left", padx=(10, 0))

message_box.bind(
    "<Control-Return>",
    lambda event: classify_message(),
)


# ============================================================
# EXAMPLE MESSAGES
# ============================================================

make_label(
    workspace, "QUICK TESTS", 9, MUTED, True
).pack(anchor="w", pady=(19, 9))

examples = tk.Frame(workspace, bg=PANEL)
examples.pack(fill="x")

make_button(
    examples,
    "Normal message",
    lambda: insert_example(
        "Hi, are we meeting at college tomorrow?"
    ),
).pack(side="left")

make_button(
    examples,
    "Prize / offer",
    lambda: insert_example(
        "Congratulations! You have won a FREE prize. "
        "Click here to claim now!!!"
    ),
).pack(side="left", padx=(9, 0))

make_button(
    examples,
    "OTP message",
    lambda: insert_example(
        "Your verification code is 482913. Do not share it."
    ),
).pack(side="left", padx=(9, 0))


# ============================================================
# SCANNING INDICATOR
# ============================================================

scanning_panel = tk.Frame(
    workspace, bg=CARD, padx=15, pady=12
)

make_label(
    scanning_panel, "ANALYZING MESSAGE PATTERNS", 9, TEAL, True
).pack(anchor="w")

make_label(
    scanning_panel,
    "Comparing text features with learned patterns...",
    9, MUTED,
).pack(anchor="w", pady=(4, 9))

scanning_canvas = tk.Canvas(
    scanning_panel, width=320, height=4,
    bg=CARD, highlightthickness=0,
)
scanning_canvas.pack(fill="x")


# ============================================================
# RESULT CARD
# ============================================================

result_panel = tk.Frame(
    workspace, bg=CARD, padx=17, pady=15,
    highlightbackground=TEAL, highlightthickness=1,
)

result_header = tk.Frame(result_panel, bg=CARD)
result_header.pack(fill="x")

result_icon = make_label(
    result_header, "✓", 24, GREEN, True
)
result_icon.pack(side="left", padx=(0, 13))

result_text = tk.Frame(result_header, bg=CARD)
result_text.pack(side="left", fill="x", expand=True)

result_title = make_label(
    result_text, "RESULT", 15, GREEN, True
)
result_title.pack(anchor="w")

result_description = make_label(
    result_text,
    "Your classification will appear here.",
    9, MUTED, wraplength=560, justify="left",
)
result_description.pack(anchor="w", pady=(4, 0))

result_badge = make_label(
    result_panel, "CLASSIFICATION", 8, GREEN, True
)
result_badge.pack(anchor="w", pady=(13, 7))

probability_row = tk.Frame(result_panel, bg=CARD)
probability_row.pack(fill="x")

make_label(
    probability_row, "ESTIMATED SPAM PROBABILITY", 9, MUTED, True
).pack(side="left")

probability_label = make_label(
    probability_row, "--", 12, WHITE, True
)
probability_label.pack(side="right")

probability_bar = tk.Canvas(
    result_panel, height=10, bg=CARD,
    highlightthickness=0,
)
probability_bar.pack(fill="x", pady=(9, 0))

make_label(
    result_panel,
    "Model estimates can be wrong. Verify suspicious links and senders.",
    8, MUTED,
).pack(anchor="w", pady=(10, 0))


# ============================================================
# FOOTER
# ============================================================

footer = tk.Frame(root, bg=BG)
footer.pack(fill="x", padx=34, pady=(0, 17))

status_label = make_label(
    footer, "Starting model training...", 9, MUTED
)
status_label.pack(anchor="w")

make_label(
    footer,
    "SMS SHIELD  •  TF-IDF + LOGISTIC REGRESSION  •  LOCAL DESKTOP APP",
    8, MUTED,
).pack(anchor="w", pady=(5, 0))


# ============================================================
# START
# ============================================================

Thread(target=train_in_background, daemon=True).start()

root.after(200, check_training_status)
root.after(100, animate_header)
root.after(70, animate_training)

root.mainloop()
