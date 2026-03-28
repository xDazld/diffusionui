import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import huggingface_hub as hf_hub
import openvino_genai as ov_genai
from PIL import Image, ImageTk

MODEL_ID = "OpenVINO/stable-diffusion-v1-5-int8-ov"
DEVICE = "CPU"
DEFAULT_STEPS = 20


class ImageGenerator:
    def __init__(
            self,
            model_id: str = MODEL_ID,
            device: str = DEVICE
            ) -> None:
        self.model_id = model_id
        self.device = device
        self.pipeline: ov_genai.Text2ImagePipeline | None = None

    def _ensure_pipeline(
            self,
            status_callback: Callable[[str], None]
            ) -> None:
        if self.pipeline is not None:
            return
        status_callback("Downloading model…")
        model_path = hf_hub.snapshot_download(self.model_id)
        status_callback("Loading pipeline…")
        self.pipeline = ov_genai.Text2ImagePipeline(model_path, self.device)
        status_callback("")

    def generate_image(
            self,
            prompt: str,
            steps: int,
            status_callback: Callable[[str], None], ) -> Image.Image:
        self._ensure_pipeline(status_callback=status_callback)
        assert self.pipeline is not None
        status_callback("Generating image…")
        image_tensor = self.pipeline.generate(prompt, num_inference_steps=steps)
        print("Generation done")
        status_callback("")
        return Image.fromarray(image_tensor.data[0])


class DiffusionUI(tk.Tk):
    def __init__(
            self
            ) -> None:
        super().__init__()
        self.title("Diffusion UI")
        self.geometry("900x900")

        self.generator = ImageGenerator()
        self.preview_photo: ImageTk.PhotoImage | None = None

        self._build_layout()

    def _build_layout(
            self
            ) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        controls = ttk.Frame(root)
        controls.pack(fill="x")

        ttk.Label(controls, text="Prompt:").grid(row=0, column=0, sticky="w")

        self.prompt_var = tk.StringVar(value="sailing ship in storm by Rembrandt")
        self.prompt_entry = ttk.Entry(controls, textvariable=self.prompt_var)
        self.prompt_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        ttk.Label(controls, text="Steps:").grid(row=0, column=2, sticky="e")
        self.steps_var = tk.IntVar(value=DEFAULT_STEPS)
        self.steps_spin = ttk.Spinbox(controls, from_=1, to=100, textvariable=self.steps_var,
            width=5, )
        self.steps_spin.grid(row=0, column=3, padx=(8, 8))

        self.generate_button = ttk.Button(controls, text="Generate", command=self._start_generation)
        self.generate_button.grid(row=0, column=4)

        controls.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", pady=(10, 8))

        preview_frame = ttk.LabelFrame(root, text="Image", padding=8)
        preview_frame.pack(fill="both", expand=True)

        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        self.prompt_entry.focus_set()
        self.bind("<Return>", lambda
            _event: self._start_generation())

    def _start_generation(
            self
            ) -> None:
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Missing prompt", "Please enter a prompt.")
            return

        steps = self.steps_var.get()
        self.generate_button.config(state=tk.DISABLED)

        # Run model loading/inference off the UI thread to keep the window responsive.
        worker = threading.Thread(target=self._generate_in_background, args=(prompt, steps),
            daemon=True, )
        worker.start()

    def _generate_in_background(
            self,
            prompt: str,
            steps: int
            ) -> None:
        try:
            pil_image = self.generator.generate_image(prompt, steps,
                status_callback=self._set_status_from_worker, )
            self.after(0, self._on_generation_success, pil_image)
        except Exception as error:  # noqa: BLE001 - surface any generation failure in the UI
            self.after(0, self._on_generation_error, error)

    def _set_status_from_worker(
            self,
            text: str
            ) -> None:
        self.after(0, self.status_var.set, text)

    def _on_generation_success(
            self,
            image: Image.Image
            ) -> None:
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.preview_photo)

        self.status_var.set("Done")
        self.generate_button.config(state=tk.NORMAL)

    def _on_generation_error(
            self,
            error: Exception
            ) -> None:
        self.status_var.set("Generation failed")
        self.generate_button.config(state=tk.NORMAL)
        messagebox.showerror("Generation failed", str(error))


def main() -> None:
    app = DiffusionUI()
    app.mainloop()


if __name__ == "__main__":
    main()
