import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk, ImageDraw
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision import transforms
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os
from datetime import datetime
import numpy as np
import io

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Assume utils and models are in your project structure or can be mocked for testing
try:
    from utils import SkinLesionDataset
    from models.cnn_transformer import EfficientNetModel
    from models.attention_model import MobileNetModel
    from models.vit_model import ViTModel
    from models.diversity_model import DiversityModel
except ImportError:
    print("Warning: Model files or utils.py not found. Some functionalities may be limited.")
    # Mock classes for GUI to run without actual models if files are missing
    class MockModel:
        def __init__(self, num_classes=39):
            pass
        def to(self, device):
            return self
        def load_state_dict(self, state_dict):
            pass
        def eval(self):
            pass
        def __call__(self, x):
            return torch.randn(1, 39)

    class EfficientNetModel(MockModel): pass
    class MobileNetModel(MockModel): pass
    class ViTModel(MockModel): pass
    class DiversityModel(MockModel):
        def __init__(self, models, weights):
            super().__init__()
            self.models = models
        def __call__(self, x):
            return torch.randn(1, 39)

class SkinLesionGUI:
    def __init__(self, root):
        self.root = root
        # === FIX: Ensure window controls are visible from the start ===
        self.root.overrideredirect(False)
        self.root.title("🔬 AI Skin Lesion Classifier - Medical Diagnosis Assistant")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#f8fafc")
        self.root.minsize(1200, 800)

        # Modern color scheme
        self.colors = {
            'primary': '#3b82f6',      # Blue
            'primary_dark': '#1e40af',  # Dark blue
            'secondary': '#10b981',      # Green
            'warning': '#f59e0b',       # Yellow
            'danger': '#ef4444',       # Red
            'background': '#f8fafc',     # Light gray
            'surface': '#ffffff',      # White
            'text_primary': '#1f2937',  # Dark gray
            'text_secondary': '#6b7280', # Medium gray
            'border': '#e5e7eb',       # Light border
            'success': '#22c55e'        # Success green
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.effnet = None
        self.mobilenet = None
        self.vit = None
        self.model = None
        self.idx2label = None
        self.model_accuracy = "N/A"
        self.dark_mode = False
        self.prediction_data = [] # Stores (rank, lesion, probability, confidence) for PDF
        self.status_text = tk.StringVar(value="🟢 Ready - Load an image to begin analysis")
        self.current_image_path = None
        self.current_image_pil = None # Store the PIL image for analysis and PDF export
        self.analysis_history = []
        self.last_probabilities = None # Store probabilities for PDF

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.setup_styles()
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        self.load_model_and_accuracy()
        
        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-s>', lambda e: self.export_to_pdf())
        self.root.bind('<F5>', lambda e: self.reload_model())

    def setup_styles(self):
        """Setup modern styling for the application"""
        self.style = ThemedStyle(self.root)
        self.style.set_theme("arc")
        
        style = ttk.Style()
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), padding=(15, 10), focuscolor='none')
        style.map("Primary.TButton", background=[("active", self.colors['primary_dark']), ("pressed", self.colors['primary_dark'])])
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(12, 8), focuscolor='none')

    def create_header(self):
        """Create modern header"""
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=100)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        header_content = tk.Frame(header_frame, bg=self.colors['primary'])
        header_content.pack(expand=True, fill="both", padx=30, pady=20)
        title_label = tk.Label(header_content, text="🔬 AI Skin Lesion Classifier", font=("Segoe UI", 28, "bold"), bg=self.colors['primary'], fg="white")
        title_label.pack(anchor="w")
        subtitle_label = tk.Label(header_content, text="Advanced Medical Diagnosis Assistant with Ensemble AI Models", font=("Segoe UI", 14), bg=self.colors['primary'], fg="#bfdbfe")
        subtitle_label.pack(anchor="w", pady=(5, 0))

    def create_main_content(self):
        """Create main content area"""
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        paned = ttk.PanedWindow(main_container, orient="horizontal")
        paned.pack(fill="both", expand=True)
        self.create_left_panel(paned)
        self.create_right_panel(paned)
    
    # === NEW HELPER FUNCTION FOR SCROLLING ===
    def _bind_mouse_scroll(self, widget, canvas):
        """Binds mouse wheel scrolling to a widget and its children."""
        widget.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', lambda event: self._on_mousewheel(event, canvas)))
        widget.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

    def _on_mousewheel(self, event, canvas):
        """Handles mouse wheel scrolling."""
        # On Windows and macOS, event.delta is used. On Linux, event.num is used.
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        else: # For Windows/macOS
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    # === END NEW HELPER ===
            
    def create_left_panel(self, parent):
        """Create a scrollable left panel."""
        left_panel_container = tk.Frame(parent, bg=self.colors['surface'], relief="solid", bd=1)
        parent.add(left_panel_container, weight=1)

        canvas = tk.Canvas(left_panel_container, bg=self.colors['surface'], highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['surface'])

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set)

        # === FIX: Use helper function for independent scrolling ===
        self._bind_mouse_scroll(left_panel_container, canvas)

        canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        
        image_section = tk.Frame(scrollable_frame, bg=self.colors['surface'])
        image_section.pack(fill="x", padx=20, pady=20, anchor="n")
        
        tk.Label(image_section, text="📷 Medical Image Analysis", font=("Segoe UI", 16, "bold"), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(anchor="w", pady=(0, 15))
        
        image_container = tk.Frame(image_section, bg=self.colors['surface'])
        image_container.pack(pady=10)
        
        self.canvas_image = tk.Canvas(image_container, width=400, height=400, bg="#f9fafb", highlightthickness=2, highlightbackground=self.colors['border'], relief="solid", bd=1)
        self.canvas_image.pack(pady=10)
        
        self.create_placeholder_image()
        self.create_control_buttons(image_section)
        
        self.create_model_info_section(scrollable_frame)
        self.create_patient_info_section(scrollable_frame)

    def create_placeholder_image(self):
        placeholder = Image.new('RGB', (400, 400), color="#f9fafb")
        draw = ImageDraw.Draw(placeholder)
        center_x, center_y = 200, 200
        draw.rectangle([center_x-40, center_y-20, center_x+40, center_y+20], outline=self.colors['border'], width=3)
        draw.text((center_x-60, center_y+40), "Click 'Load Image' to start", fill=self.colors['text_secondary'])
        self.placeholder_img = ImageTk.PhotoImage(placeholder)
        self.canvas_image.create_image(200, 200, image=self.placeholder_img, anchor=tk.CENTER)

    def create_control_buttons(self, parent):
        button_frame = tk.Frame(parent, bg=self.colors['surface'])
        button_frame.pack(fill="x", pady=20)
        primary_frame = tk.Frame(button_frame, bg=self.colors['surface'])
        primary_frame.pack(fill="x", pady=(0, 10))
        self.load_btn = ttk.Button(primary_frame, text="📷 Load Image", command=self.load_image, style="Primary.TButton")
        self.load_btn.pack(side="left", padx=(0, 10))
        self.analyze_btn = ttk.Button(primary_frame, text="🔍 Analyze Lesion", command=self.run_analysis, style="Primary.TButton", state="disabled")
        self.analyze_btn.pack(side="left", padx=(0, 10))
        secondary_frame = tk.Frame(button_frame, bg=self.colors['surface'])
        secondary_frame.pack(fill="x")
        self.clear_btn = ttk.Button(secondary_frame, text="🗑️ Clear", command=self.clear_image, style="Secondary.TButton")
        self.clear_btn.pack(side="left", padx=(0, 5))
        self.reload_btn = ttk.Button(secondary_frame, text="🔄 Reload Model", command=self.reload_model, style="Secondary.TButton")
        self.reload_btn.pack(side="left", padx=(0, 5))
        self.export_btn = ttk.Button(secondary_frame, text="📄 Export PDF", command=self.export_to_pdf, style="Secondary.TButton")
        self.export_btn.pack(side="left", padx=(0, 5))
        self.dark_toggle = ttk.Button(secondary_frame, text="🌓 Dark Mode", command=self.toggle_theme, style="Secondary.TButton")
        self.dark_toggle.pack(side="left")

    def create_model_info_section(self, parent):
        info_section = tk.Frame(parent, bg=self.colors['surface'])
        info_section.pack(fill="x", padx=20, pady=(0, 20), anchor="n")
        tk.Label(info_section, text="🤖 Model Information", font=("Segoe UI", 14, "bold"), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(anchor="w", pady=(0, 10))
        self.model_info_frame = tk.Frame(info_section, bg="#e0f2fe", relief="solid", bd=1)
        self.model_info_frame.pack(fill="x", pady=5)
        tk.Label(self.model_info_frame, text="Ensemble Model Accuracy:", font=("Segoe UI", 11, "bold"), bg="#e0f2fe", fg=self.colors['text_primary']).pack(anchor="w", padx=10, pady=(10, 5))
        self.model_accuracy_label = tk.Label(self.model_info_frame, text=self.model_accuracy, font=("Segoe UI", 20, "bold"), bg="#e0f2fe", fg=self.colors['success'])
        self.model_accuracy_label.pack(anchor="w", padx=10, pady=(0, 10))

    def create_patient_info_section(self, parent):
        info_section = tk.Frame(parent, bg=self.colors['surface'])
        info_section.pack(fill="x", padx=20, pady=(0, 20), anchor="n")
        tk.Label(info_section, text="📋 Patient Information", font=("Segoe UI", 14, "bold"), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(anchor="w", pady=(0, 10))
        name_frame = tk.Frame(info_section, bg=self.colors['surface'])
        name_frame.pack(fill="x", pady=5)
        tk.Label(name_frame, text="Name:", font=("Segoe UI", 11), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(side="left")
        self.name_entry = tk.Entry(name_frame, font=("Segoe UI", 11), width=30)
        self.name_entry.pack(side="left", padx=5)
        age_frame = tk.Frame(info_section, bg=self.colors['surface'])
        age_frame.pack(fill="x", pady=5)
        tk.Label(age_frame, text="Age:", font=("Segoe UI", 11), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(side="left")
        self.age_entry = tk.Entry(age_frame, font=("Segoe UI", 11), width=10)
        self.age_entry.pack(side="left", padx=5)
        gender_frame = tk.Frame(info_section, bg=self.colors['surface'])
        gender_frame.pack(fill="x", pady=5)
        tk.Label(gender_frame, text="Gender:", font=("Segoe UI", 11), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(side="left")
        self.gender_var = tk.StringVar(value="Not Specified")
        gender_options = ["Male", "Female", "Other", "Not Specified"]
        gender_menu = ttk.OptionMenu(gender_frame, self.gender_var, "Not Specified", *gender_options)
        gender_menu.config(width=15)
        gender_menu.pack(side="left", padx=5)

    def create_right_panel(self, parent):
        right_frame = tk.Frame(parent, bg=self.colors['surface'], relief="solid", bd=1)
        parent.add(right_frame, weight=2)
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.create_results_tab(notebook)
        self.create_analysis_tab(notebook)
        self.create_history_tab(notebook)

    def create_results_tab(self, notebook):
        results_tab_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(results_tab_frame, text="📊 Results")

        # === Create a scrollable area for the results tab with HORIZONTAL scroll ===
        canvas = tk.Canvas(results_tab_frame, bg=self.colors['surface'], highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(results_tab_frame, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(results_tab_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['surface'])

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # === FIX: Use helper function for independent scrolling ===
        self._bind_mouse_scroll(results_tab_frame, canvas)

        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        # === END SCROLLABLE AREA SETUP ===
        
        results_section = tk.Frame(scrollable_frame, bg=self.colors['surface'])
        results_section.pack(fill="x", padx=20, pady=20)
        
        tk.Label(results_section, text="🎯 Diagnosis Results", font=("Segoe UI", 18, "bold"), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(anchor="w", pady=(0, 15))
        self.prediction_frame = tk.Frame(results_section, bg="#f0f9ff", relief="solid", bd=2)
        self.prediction_frame.pack(fill="x", pady=10)
        self.prediction_label = tk.Label(self.prediction_frame, text="🔬 Predicted Lesion: Ready for analysis", font=("Segoe UI", 16, "bold"), bg="#f0f9ff", fg=self.colors['primary'])
        self.prediction_label.pack(anchor="w", padx=15, pady=(15, 5))
        self.confidence_label = tk.Label(self.prediction_frame, text="📈 Confidence: Awaiting image", font=("Segoe UI", 14), bg="#f0f9ff", fg=self.colors['text_secondary'])
        self.confidence_label.pack(anchor="w", padx=15, pady=(0, 15))
        
        viz_section = tk.Frame(scrollable_frame, bg=self.colors['surface'])
        viz_section.pack(fill="x", expand=True, padx=20, pady=(0, 20))
        
        tk.Label(viz_section, text="📈 Probability Distribution", font=("Segoe UI", 16, "bold"), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(anchor="w", pady=(0, 10))
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.patch.set_facecolor('white')
        self.canvas_vis = FigureCanvasTkAgg(self.fig, master=viz_section)
        self.canvas_vis.get_tk_widget().pack(fill="both", expand=True)
        self.setup_empty_plot()

    # The rest of the functions (create_analysis_tab, create_history_tab, etc.) remain unchanged
    # as their scrolling is handled by their respective widgets (Treeview, Listbox) or was already correct.

    def create_analysis_tab(self, notebook):
        """Create detailed analysis tab"""
        analysis_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(analysis_frame, text="🔍 Detailed Analysis")
        canvas = tk.Canvas(analysis_frame, bg=self.colors['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['surface'])
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        self._bind_mouse_scroll(analysis_frame, canvas)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        table_header = tk.Frame(scrollable_frame, bg=self.colors['surface'])
        table_header.pack(fill="x", padx=20, pady=20)
        tk.Label(table_header, text="📋 Complete Analysis - All 39 Skin Lesion Types", font=("Segoe UI", 16, "bold"), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(anchor="w")
        table_container = tk.Frame(scrollable_frame, bg=self.colors['surface'])
        table_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.table = ttk.Treeview(table_container, columns=("rank", "lesion", "probability", "confidence"), show="headings", height=20)
        self.table.heading("rank", text="Rank")
        self.table.heading("lesion", text="Lesion Type")
        self.table.heading("probability", text="Probability (%)")
        self.table.heading("confidence", text="Confidence Level")
        self.table.column("rank", width=70, anchor="center")
        self.table.column("lesion", width=350, anchor="w")
        self.table.column("probability", width=120, anchor="center")
        self.table.column("confidence", width=120, anchor="center")
        self.table.pack(side="left", fill="both", expand=True)
        table_scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=table_scrollbar.set)
        table_scrollbar.pack(side="right", fill="y")
        self.table.tag_configure('high', background='#dcfce7', foreground='#166534')
        self.table.tag_configure('medium', background='#fef3c7', foreground='#92400e')
        self.table.tag_configure('low', background='#fee2e2', foreground='#991b1b')
        self.table.tag_configure('normal', background='white', foreground='#374151')

    def create_history_tab(self, notebook):
        """Create analysis history tab"""
        history_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(history_frame, text="📚 History")
        tk.Label(history_frame, text="📚 Analysis History", font=("Segoe UI", 16, "bold"), bg=self.colors['surface'], fg=self.colors['text_primary']).pack(anchor="w", padx=20, pady=20)
        history_list_frame = tk.Frame(history_frame)
        history_list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.history_listbox = tk.Listbox(history_list_frame, font=("Segoe UI", 11), bg=self.colors['surface'], selectbackground=self.colors['primary'], selectforeground="white")
        history_scrollbar = ttk.Scrollbar(history_list_frame, orient="vertical", command=self.history_listbox.yview)
        self.history_listbox.config(yscrollcommand=history_scrollbar.set)
        history_scrollbar.pack(side="right", fill="y")
        self.history_listbox.pack(side="left", fill="both", expand=True)

    def setup_empty_plot(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, '📊 Load an image and click Analyze to see the probability distribution',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax.transAxes, fontsize=14,
                     color=self.colors['text_secondary'])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas_vis.draw()

    def create_status_bar(self):
        status_frame = tk.Frame(self.root, bg=self.colors['primary_dark'], height=30)
        status_frame.pack(side="bottom", fill="x")
        status_frame.pack_propagate(False)
        self.status_label = tk.Label(status_frame, textvariable=self.status_text, font=("Segoe UI", 10), bg=self.colors['primary_dark'], fg="white")
        self.status_label.pack(side="left", padx=20, pady=5)
        device_text = f"🖥️ Device: {self.device.type.upper()}"
        device_label = tk.Label(status_frame, text=device_text, font=("Segoe UI", 10), bg=self.colors['primary_dark'], fg="#bfdbfe")
        device_label.pack(side="right", padx=20, pady=5)

    def load_model_and_accuracy(self):
        self.status_text.set("🔄 Loading AI models...")
        self.root.update()
        required_files = ["output/label_mapping.json", "output/label_fullnames.json", "output/div_model.pth", "output/effnet_pretrained.pth", "output/mobilenet_pretrained.pth", "output/vit_pretrained.pth"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            self.status_text.set("❌ Error: Missing model files")
            messagebox.showerror("Missing Files", f"The following required files are missing:\n\n" + "\n".join(missing_files) + "\n\nPlease ensure all model files are in the 'output' directory.")
            return
        try:
            with open("output/label_mapping.json", "r") as f:
                self.idx2label = {int(k): v for k, v in json.load(f).items()}
            num_classes = len(self.idx2label)
            self.effnet = EfficientNetModel(num_classes).to(self.device)
            self.effnet.load_state_dict(torch.load("output/effnet_pretrained.pth", map_location=self.device))
            self.effnet.eval()
            self.mobilenet = MobileNetModel(num_classes).to(self.device)
            self.mobilenet.load_state_dict(torch.load("output/mobilenet_pretrained.pth", map_location=self.device))
            self.mobilenet.eval()
            self.vit = ViTModel(num_classes).to(self.device)
            self.vit.load_state_dict(torch.load("output/vit_pretrained.pth", map_location=self.device))
            self.vit.eval()
            self.model = DiversityModel([self.effnet, self.mobilenet, self.vit], weights=[0.33, 0.33, 0.33]).to(self.device)
            self.model.load_state_dict(torch.load("output/div_model.pth", map_location=self.device))
            self.model.eval()
            accuracy_file = "output/last_accuracy.txt"
            if os.path.exists(accuracy_file):
                with open(accuracy_file, "r") as f:
                    accuracy_str = f.read().strip()
                    self.model_accuracy = f"{float(accuracy_str.replace('%', '')):.2f}%" if '%' in accuracy_str else f"{float(accuracy_str) * 100:.2f}%"
            else:
                self.model_accuracy = "N/A"
            self.model_accuracy_label.config(text=self.model_accuracy)
            self.status_text.set("✅ All models loaded successfully - Ready for analysis")
        except Exception as e:
            self.model = None
            self.status_text.set(f"❌ Error loading models: {str(e)}")
            messagebox.showerror("Model Loading Error", f"Failed to load models:\n\n{str(e)}\n\nPlease check the model files and try again.")

    def load_image(self):
        if not self.model:
            messagebox.showerror("Model Error", "AI models not loaded. Please reload the models first.")
            return
        self.status_text.set("📁 Opening file dialog...")
        file_path = filedialog.askopenfilename(title="Select Medical Image", filetypes=[("All Image Files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), ("JPEG Files", "*.jpg *.jpeg"), ("PNG Files", "*.png"), ("All Files", "*.*")])
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state="normal")
            self.status_text.set("✅ Image loaded. Click 'Analyze Lesion' to proceed.")
        else:
            self.status_text.set("🟡 No image selected")

    def display_image(self, file_path):
        try:
            self.clear_results()
            img = Image.open(file_path).convert("RGB")
            self.current_image_pil = img.copy()
            img_for_display = img.copy()
            img_for_display.thumbnail((400, 400), Image.Resampling.LANCZOS)
            bg = Image.new('RGB', (400, 400), color="#f9fafb")
            offset = ((400 - img_for_display.width) // 2, (400 - img_for_display.height) // 2)
            bg.paste(img_for_display, offset)
            photo = ImageTk.PhotoImage(bg)
            self.canvas_image.delete("all")
            self.canvas_image.create_image(200, 200, image=photo, anchor=tk.CENTER)
            self.canvas_image.image = photo
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to load or display the image:\n\n{e}")
            self.status_text.set(f"❌ Error loading image: {e}")

    def run_analysis(self):
        if self.current_image_pil is None:
            messagebox.showwarning("No Image", "Please load an image before analyzing.")
            return
        try:
            self.status_text.set("🤖 Running AI analysis...")
            self.root.update()
            img_tensor = self.transform(self.current_image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                ensemble_logits = self.model(img_tensor)
                ensemble_probs = torch.nn.functional.softmax(ensemble_logits, dim=1)[0]
                self.last_probabilities = ensemble_probs.cpu().numpy()
                ensemble_pred_idx = ensemble_probs.argmax().item()
                ensemble_confidence = ensemble_probs[ensemble_pred_idx].item()
            with open("output/label_fullnames.json", "r") as f:
                label_fullnames = json.load(f)
            label = self.idx2label[ensemble_pred_idx]
            full_label = label_fullnames.get(label, label)
            self.prediction_label.config(text=f"🔬 Predicted Lesion: {full_label}")
            confidence_color = self.get_confidence_color(ensemble_confidence)
            self.confidence_label.config(text=f"📈 Confidence: {ensemble_confidence * 100:.2f}%", fg=confidence_color)
            self.update_visualization(self.last_probabilities, label_fullnames)
            self.update_analysis_table(self.last_probabilities, label_fullnames)
            self.add_to_history(full_label, ensemble_confidence, self.current_image_path)
            self.status_text.set("✅ Analysis complete")
        except Exception as e:
            self.status_text.set(f"❌ Analysis error: {str(e)}")
            messagebox.showerror("Analysis Error", f"An unexpected error occurred during analysis:\n\n{str(e)}")

    def get_confidence_color(self, confidence):
        if confidence >= 0.8: return self.colors['success']
        elif confidence >= 0.6: return self.colors['warning']
        else: return self.colors['danger']

    def update_visualization(self, probabilities, label_fullnames):
        self.ax.clear()
        top_indices = np.argsort(probabilities)[-10:][::-1]
        top_probs = probabilities[top_indices] * 100
        top_labels = [label_fullnames.get(self.idx2label[i], self.idx2label[i]) for i in top_indices]
        top_labels = [label[:30] + "..." if len(label) > 30 else label for label in top_labels]
        bars = self.ax.barh(range(len(top_labels)), top_probs, color=[self.colors['primary'] if i == 0 else self.colors['text_secondary'] for i in range(len(top_labels))])
        self.ax.set_yticks(range(len(top_labels)))
        self.ax.set_yticklabels(top_labels, fontsize=11, color=self.colors['text_primary'])
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Confidence (%)", fontsize=12, color=self.colors['text_primary'])
        self.ax.set_title("Top 10 Predictions", fontsize=14, fontweight='bold', color=self.colors['text_primary'], pad=20)
        for bar, prob in zip(bars, top_probs):
            if prob > 0.5:
                self.ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{prob:.1f}%", va='center', fontsize=10, color=self.colors['text_primary'])
        self.fig.tight_layout()
        self.canvas_vis.draw()

    def update_analysis_table(self, probabilities, label_fullnames):
        for row in self.table.get_children(): self.table.delete(row)
        self.prediction_data = []
        sorted_indices = np.argsort(probabilities)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            label = self.idx2label[idx]
            full_label = label_fullnames.get(label, label)
            confidence = probabilities[idx] * 100
            if confidence >= 80: confidence_level, tag = "High", 'high'
            elif confidence >= 60: confidence_level, tag = "Medium", 'medium'
            elif confidence >= 10: confidence_level, tag = "Low", 'low'
            else: confidence_level, tag = "Very Low", 'normal'
            self.table.insert("", "end", values=(rank, full_label, f"{confidence:.2f}", confidence_level), tags=(tag,))
            self.prediction_data.append((rank, full_label, f"{confidence:.2f}%", confidence_level))

    def add_to_history(self, prediction, confidence, file_path):
        entry_text = f"{prediction} ({confidence*100:.2f}%) - {os.path.basename(file_path)} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.history_listbox.insert(0, entry_text)
        self.analysis_history.insert(0, {"prediction": prediction, "confidence": confidence, "file_path": file_path, "time": datetime.now()})

    def export_to_pdf(self):
        if self.current_image_path is None or self.last_probabilities is None:
            messagebox.showwarning("Export Failed", "No analysis data to export. Please load and analyze an image first.")
            return
        pdf_filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")], title="Save PDF Report", initialfile=f"SkinLesion_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        if not pdf_filename:
            self.status_text.set("🟡 PDF export canceled")
            return
        self.status_text.set("📤 Generating PDF report...")
        self.root.update()
        try:
            doc = SimpleDocTemplate(pdf_filename, pagesize=A4, rightMargin=inch/2, leftMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
            story, styles = [], getSampleStyleSheet()
            story.append(Paragraph("AI Skin Lesion Analysis Report", styles['h1']))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Patient Information", styles['h2']))
            patient_info_text = f"<b>Name:</b> {self.name_entry.get() or 'N/A'}<br/><b>Age:</b> {self.age_entry.get() or 'N/A'}<br/><b>Gender:</b> {self.gender_var.get()}"
            story.append(Paragraph(patient_info_text, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Analysis Summary", styles['h2']))
            img_report = RLImage(self.current_image_path, width=2.5*inch, height=2.5*inch)
            img_report.hAlign = 'CENTER'
            buffer = io.BytesIO()
            self.fig.savefig(buffer, format='png', dpi=150, bbox_inches="tight")
            buffer.seek(0)
            chart_report = RLImage(buffer, width=5*inch, height=3.5*inch)
            summary_table = Table([[img_report, chart_report]], colWidths=[2.7*inch, 5.2*inch])
            summary_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0,0),(-1,-1),0), ('RIGHTPADDING',(0,0),(-1,-1),0)]))
            story.append(summary_table)
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Detailed Prediction Results", styles['h2']))
            table_data = [["Rank", "Lesion Type", "Probability", "Confidence Level"]] + self.prediction_data
            detailed_table = Table(table_data, colWidths=[0.5*inch, 4.0*inch, 1.2*inch, 1.5*inch])
            detailed_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('ALIGN', (1, 1), (1, -1), 'LEFT'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.beige), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
            story.append(detailed_table)
            story.append(Spacer(1, 0.3*inch))
            disclaimer_text = ("<i><b>Disclaimer:</b> This is an AI-generated analysis intended for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.</i>")
            story.append(Paragraph(disclaimer_text, styles['Italic']))
            doc.build(story)
            self.status_text.set("✅ PDF report exported successfully")
            messagebox.showinfo("Export Successful", f"PDF report saved as {pdf_filename}")
        except Exception as e:
            self.status_text.set(f"❌ Error exporting PDF: {e}")
            messagebox.showerror("Export Failed", f"An unexpected error occurred while creating the PDF:\n\n{e}")

    def clear_results(self):
        self.prediction_label.config(text="🔬 Predicted Lesion: Ready for analysis")
        self.confidence_label.config(text="📈 Confidence: Awaiting image", fg=self.colors['text_secondary'])
        self.setup_empty_plot()
        for item in self.table.get_children(): self.table.delete(item)
        self.prediction_data, self.last_probabilities = [], None

    def clear_image(self):
        if not messagebox.askyesno("Clear Analysis", "Are you sure you want to clear the current analysis?"): return
        self.status_text.set("🧹 Clearing analysis...")
        self.canvas_image.delete("all")
        self.create_placeholder_image()
        self.clear_results()
        self.current_image_path, self.current_image_pil = None, None
        self.analyze_btn.config(state="disabled")
        self.status_text.set("✅ Analysis cleared - Ready for new image")

    def reload_model(self):
        if messagebox.askyesno("Reload Models", "This will reload all AI models from disk.\n\nDo you want to continue?"):
            self.status_text.set("🔄 Reloading AI models...")
            self.effnet, self.mobilenet, self.vit, self.model = None, None, None, None
            self.load_model_and_accuracy()
            messagebox.showinfo("Reload Complete", "All AI models have been reloaded successfully!")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        theme = "equilux" if self.dark_mode else "arc"
        self.style.set_theme(theme)
        if self.dark_mode:
            self.colors.update({'background': "#2d3748", 'surface': "#4a5568", 'text_primary': "#f7fafc", 'text_secondary': "#e2e8f0"})
        else:
            self.colors.update({'background': "#f8fafc", 'surface': "#ffffff", 'text_primary': "#1f2937", 'text_secondary': "#6b7280"})
        self.root.configure(bg=self.colors['background'])
        # Update all existing frames/labels/buttons
        for widget in self.root.winfo_children():
            self.update_widget_colors(widget)
        self.status_text.set(f"🌙 Switched to {'dark' if self.dark_mode else 'light'} theme")

    def update_widget_colors(self, widget):
        try:
            if isinstance(widget, (tk.Frame, tk.Label, tk.Canvas)):
                widget.configure(bg=self.colors.get('surface', widget['bg']))
            if isinstance(widget, tk.Label):
                widget.configure(fg=self.colors.get('text_primary', widget['fg']))
            if isinstance(widget, tk.Button):
                widget.configure(bg=self.colors.get('primary'), fg="white")
        except:
            pass
        for child in widget.winfo_children():
            self.update_widget_colors(child)

    


    def show_about(self):
        about_text = """
🔬 AI Skin Lesion Classifier v2.0
Developed with PyTorch and Modern UI Design

Features:
• Ensemble AI (EfficientNet + MobileNet + ViT)
• Classifies 39 different skin lesion types
• Detailed probability distributions & PDF reporting

⚠️ MEDICAL DISCLAIMER:
This software is for research and educational purposes only.
It is NOT a substitute for professional medical advice.
        """
        messagebox.showinfo("About", about_text)

def add_menu_bar(root, app):
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Load Image (Ctrl+O)", command=app.load_image)
    file_menu.add_command(label="Export PDF (Ctrl+S)", command=app.export_to_pdf)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    models_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Models", menu=models_menu)
    models_menu.add_command(label="Reload Models (F5)", command=app.reload_model)
    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Toggle Dark Mode", command=app.toggle_theme)
    view_menu.add_command(label="Clear Analysis", command=app.clear_image)
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=app.show_about)

if __name__ == "__main__":
    root = tk.Tk()
    app = SkinLesionGUI(root)
    add_menu_bar(root, app)
    
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()