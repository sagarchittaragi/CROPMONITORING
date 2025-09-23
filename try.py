import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import random
import json


class AdvancedPlantAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Plant Health, Species & Risk Analyzer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Plant database (in a real app, this would be from a trained model)
        self.plant_database = {
            "tomato": {
                "scientific_name": "Solanum lycopersicum",
                "family": "Solanaceae",
                "ideal_conditions": {
                    "temperature": "18-27°C",
                    "sunlight": "6-8 hours daily",
                    "water": "Regular, consistent moisture",
                    "soil_ph": "6.0-6.8"
                },
                "common_pests": ["Aphids", "Tomato Hornworm", "Whiteflies", "Spider Mites"],
                "common_diseases": ["Early Blight", "Late Blight", "Blossom End Rot"]
            },
            "rose": {
                "scientific_name": "Rosa spp.",
                "family": "Rosaceae",
                "ideal_conditions": {
                    "temperature": "15-25°C",
                    "sunlight": "6+ hours daily",
                    "water": "Deep watering 2-3 times weekly",
                    "soil_ph": "6.0-6.5"
                },
                "common_pests": ["Aphids", "Spider Mites", "Japanese Beetles", "Thrips"],
                "common_diseases": ["Black Spot", "Powdery Mildew", "Rust"]
            },
            "cactus": {
                "scientific_name": "Cactaceae family",
                "family": "Cactaceae",
                "ideal_conditions": {
                    "temperature": "21-35°C",
                    "sunlight": "Direct sunlight",
                    "water": "Sparse, allow soil to dry completely",
                    "soil_ph": "5.5-7.0"
                },
                "common_pests": ["Mealybugs", "Scale Insects", "Spider Mites"],
                "common_diseases": ["Root Rot", "Fungal Infections"]
            },
            "sunflower": {
                "scientific_name": "Helianthus annuus",
                "family": "Asteraceae",
                "ideal_conditions": {
                    "temperature": "18-30°C",
                    "sunlight": "6-8 hours direct sun",
                    "water": "Regular, drought-tolerant once established",
                    "soil_ph": "6.0-7.5"
                },
                "common_pests": ["Aphids", "Caterpillars", "Birds"],
                "common_diseases": ["Downy Mildew", "Rust", "Powdery Mildew"]
            },
            "orchid": {
                "scientific_name": "Orchidaceae family",
                "family": "Orchidaceae",
                "ideal_conditions": {
                    "temperature": "18-27°C",
                    "sunlight": "Bright indirect light",
                    "water": "Weekly, allow to dry between",
                    "soil_ph": "5.5-6.5"
                },
                "common_pests": ["Mealybugs", "Scale", "Spider Mites"],
                "common_diseases": ["Crown Rot", "Bacterial Brown Spot"]
            }
        }

        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.identified_plant = "Unknown"
        self.identification_confidence = 0
        self.plant_health_status = "Unknown"
        self.plant_health_score = 0
        self.soil_condition = "Unknown"
        self.soil_score = 0
        self.pest_risk = "Unknown"
        self.pest_risk_score = 0

        self.setup_ui()

    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="AI Plant Health, Species & Risk Analyzer",
                               font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#2d5a27')
        title_label.pack(pady=10)

        # Description
        desc_label = tk.Label(self.root,
                              text="Upload a plant image to identify species and analyze health, soil condition, and pest risks",
                              font=('Arial', 11), bg='#f0f0f0', fg='#555555')
        desc_label.pack(pady=5)

        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Left panel - Image display
        left_frame = tk.Frame(main_frame, bg='#ffffff', relief='solid', bd=1)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Image display label
        self.image_label = tk.Label(left_frame, text="No image selected",
                                    bg='#ffffff', fg='#888888', font=('Arial', 14))
        self.image_label.pack(expand=True, fill='both', padx=10, pady=10)

        # Right panel - Controls and results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0', width=500)
        right_frame.pack(side='right', fill='y', padx=(10, 0))
        right_frame.pack_propagate(False)

        # Controls frame
        controls_frame = tk.Frame(right_frame, bg='#f0f0f0')
        controls_frame.pack(fill='x', pady=(0, 15))

        # Upload button
        upload_btn = tk.Button(controls_frame, text="Upload Plant Image",
                               command=self.upload_image, bg='#4CAF50', fg='white',
                               font=('Arial', 11, 'bold'), padx=15, pady=8)
        upload_btn.pack(fill='x', pady=3)

        # Capture button (simulated)
        capture_btn = tk.Button(controls_frame, text="Simulate Camera Capture",
                                command=self.simulate_capture, bg='#2196F3', fg='white',
                                font=('Arial', 11), padx=15, pady=8)
        capture_btn.pack(fill='x', pady=3)

        # Analyze button
        analyze_btn = tk.Button(controls_frame, text="Analyze Plant",
                                command=self.analyze_plant, bg='#FF9800', fg='white',
                                font=('Arial', 11, 'bold'), padx=15, pady=8)
        analyze_btn.pack(fill='x', pady=3)

        # Results frame
        results_frame = tk.Frame(right_frame, bg='#f0f0f0')
        results_frame.pack(fill='both', expand=True)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill='both', expand=True, pady=5)

        # Plant Identification Tab
        id_tab = ttk.Frame(self.notebook)
        self.notebook.add(id_tab, text="Plant ID")
        self.setup_identification_tab(id_tab)

        # Plant Health Tab
        plant_tab = ttk.Frame(self.notebook)
        self.notebook.add(plant_tab, text="Plant Health")
        self.setup_plant_tab(plant_tab)

        # Soil Condition Tab
        soil_tab = ttk.Frame(self.notebook)
        self.notebook.add(soil_tab, text="Soil Condition")
        self.setup_soil_tab(soil_tab)

        # Pest Risk Tab
        pest_tab = ttk.Frame(self.notebook)
        self.notebook.add(pest_tab, text="Pest Risk")
        self.setup_pest_tab(pest_tab)

        # Recommendations Tab
        rec_tab = ttk.Frame(self.notebook)
        self.notebook.add(rec_tab, text="Recommendations")
        self.setup_recommendations_tab(rec_tab)

    def setup_identification_tab(self, parent):
        # Plant identification results
        id_frame = tk.Frame(parent, bg='#f0f0f0')
        id_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Identified plant
        plant_frame = tk.Frame(id_frame, bg='#f0f0f0')
        plant_frame.pack(fill='x', pady=5)

        plant_label = tk.Label(plant_frame, text="Identified Plant:",
                               font=('Arial', 12, 'bold'), bg='#f0f0f0')
        plant_label.pack(side='left')

        self.plant_value = tk.Label(plant_frame, text="Unknown",
                                    font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#2d5a27')
        self.plant_value.pack(side='left', padx=(10, 0))

        # Confidence
        conf_frame = tk.Frame(id_frame, bg='#f0f0f0')
        conf_frame.pack(fill='x', pady=5)

        conf_label = tk.Label(conf_frame, text="Confidence:",
                              font=('Arial', 12, 'bold'), bg='#f0f0f0')
        conf_label.pack(side='left')

        self.conf_value = tk.Label(conf_frame, text="0%",
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.conf_value.pack(side='left', padx=(10, 0))

        # Plant info frame
        info_frame = tk.LabelFrame(id_frame, text="Plant Information",
                                   font=('Arial', 11, 'bold'), bg='#f0f0f0')
        info_frame.pack(fill='both', expand=True, pady=10)

        self.plant_info_text = tk.Text(info_frame, wrap='word', font=('Arial', 10),
                                       bg='#ffffff', relief='solid', bd=1, height=12)
        self.plant_info_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Initial message
        self.plant_info_text.insert(tk.END, "Plant information will appear here after analysis.")
        self.plant_info_text.config(state='disabled')

    def setup_plant_tab(self, parent):
        # Health status
        status_frame = tk.Frame(parent, bg='#f0f0f0')
        status_frame.pack(fill='x', padx=10, pady=10)

        status_label = tk.Label(status_frame, text="Plant Health Status:",
                                font=('Arial', 12, 'bold'), bg='#f0f0f0')
        status_label.pack(side='left')

        self.plant_status_value = tk.Label(status_frame, text="Unknown",
                                           font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#FF9800')
        self.plant_status_value.pack(side='left', padx=(10, 0))

        # Health score
        score_frame = tk.Frame(parent, bg='#f0f0f0')
        score_frame.pack(fill='x', padx=10, pady=5)

        score_label = tk.Label(score_frame, text="Health Score:",
                               font=('Arial', 12, 'bold'), bg='#f0f0f0')
        score_label.pack(side='left')

        self.plant_score_value = tk.Label(score_frame, text="0%",
                                          font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.plant_score_value.pack(side='left', padx=(10, 0))

        # Health indicator
        indicator_frame = tk.Frame(parent, bg='#f0f0f0')
        indicator_frame.pack(fill='x', padx=10, pady=10)

        self.plant_health_indicator = tk.Label(indicator_frame, text="⬤",
                                               font=('Arial', 24), bg='#f0f0f0', fg='#cccccc')
        self.plant_health_indicator.pack()

        # Detailed analysis
        details_frame = tk.Frame(parent, bg='#f0f0f0')
        details_frame.pack(fill='both', expand=True, padx=10, pady=10)

        details_label = tk.Label(details_frame, text="Plant Health Analysis:",
                                 font=('Arial', 12, 'bold'), bg='#f0f0f0')
        details_label.pack(anchor='w')

        self.plant_details_text = tk.Text(details_frame, height=10, wrap='word',
                                          font=('Arial', 10), bg='#ffffff', relief='solid', bd=1)
        self.plant_details_text.pack(fill='both', expand=True, pady=(5, 0))

    def setup_soil_tab(self, parent):
        # Soil condition
        status_frame = tk.Frame(parent, bg='#f0f0f0')
        status_frame.pack(fill='x', padx=10, pady=10)

        status_label = tk.Label(status_frame, text="Soil Condition:",
                                font=('Arial', 12, 'bold'), bg='#f0f0f0')
        status_label.pack(side='left')

        self.soil_status_value = tk.Label(status_frame, text="Unknown",
                                          font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#FF9800')
        self.soil_status_value.pack(side='left', padx=(10, 0))

        # Soil score
        score_frame = tk.Frame(parent, bg='#f0f0f0')
        score_frame.pack(fill='x', padx=10, pady=5)

        score_label = tk.Label(score_frame, text="Soil Quality Score:",
                               font=('Arial', 12, 'bold'), bg='#f0f0f0')
        score_label.pack(side='left')

        self.soil_score_value = tk.Label(score_frame, text="0%",
                                         font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.soil_score_value.pack(side='left', padx=(10, 0))

        # Soil parameters frame
        params_frame = tk.LabelFrame(parent, text="Soil Parameters", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        params_frame.pack(fill='x', padx=10, pady=10)

        # Nutrient levels
        self.nutrient_frame = self.create_parameter_frame(params_frame, "Nutrient Levels", 0)
        self.moisture_frame = self.create_parameter_frame(params_frame, "Moisture Content", 1)
        self.ph_frame = self.create_parameter_frame(params_frame, "pH Level", 2)

        # Detailed analysis
        details_frame = tk.Frame(parent, bg='#f0f0f0')
        details_frame.pack(fill='both', expand=True, padx=10, pady=10)

        details_label = tk.Label(details_frame, text="Soil Analysis:",
                                 font=('Arial', 12, 'bold'), bg='#f0f0f0')
        details_label.pack(anchor='w')

        self.soil_details_text = tk.Text(details_frame, height=6, wrap='word',
                                         font=('Arial', 10), bg='#ffffff', relief='solid', bd=1)
        self.soil_details_text.pack(fill='both', expand=True, pady=(5, 0))

    def setup_pest_tab(self, parent):
        # Pest risk
        status_frame = tk.Frame(parent, bg='#f0f0f0')
        status_frame.pack(fill='x', padx=10, pady=10)

        status_label = tk.Label(status_frame, text="Pest Risk Level:",
                                font=('Arial', 12, 'bold'), bg='#f0f0f0')
        status_label.pack(side='left')

        self.pest_status_value = tk.Label(status_frame, text="Unknown",
                                          font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#FF9800')
        self.pest_status_value.pack(side='left', padx=(10, 0))

        # Pest risk score
        score_frame = tk.Frame(parent, bg='#f0f0f0')
        score_frame.pack(fill='x', padx=10, pady=5)

        score_label = tk.Label(score_frame, text="Risk Score:",
                               font=('Arial', 12, 'bold'), bg='#f0f0f0')
        score_label.pack(side='left')

        self.pest_score_value = tk.Label(score_frame, text="0%",
                                         font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.pest_score_value.pack(side='left', padx=(10, 0))

        # Pest detection frame
        detection_frame = tk.LabelFrame(parent, text="Pest Detection", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        detection_frame.pack(fill='x', padx=10, pady=10)

        # Common pests
        pests = ["Aphids", "Spider Mites", "Whiteflies", "Caterpillars", "Fungal Infection"]
        self.pest_vars = {}

        for i, pest in enumerate(pests):
            frame = tk.Frame(detection_frame, bg='#f0f0f0')
            frame.pack(fill='x', padx=10, pady=2)

            label = tk.Label(frame, text=pest, font=('Arial', 10), bg='#f0f0f0', width=15, anchor='w')
            label.pack(side='left')

            var = tk.StringVar(value="Not Detected")
            self.pest_vars[pest] = var

            status_label = tk.Label(frame, textvariable=var, font=('Arial', 10), bg='#f0f0f0', fg='#888888')
            status_label.pack(side='left', padx=(10, 0))

        # Detailed analysis
        details_frame = tk.Frame(parent, bg='#f0f0f0')
        details_frame.pack(fill='both', expand=True, padx=10, pady=10)

        details_label = tk.Label(details_frame, text="Pest Risk Analysis:",
                                 font=('Arial', 12, 'bold'), bg='#f0f0f0')
        details_label.pack(anchor='w')

        self.pest_details_text = tk.Text(details_frame, height=6, wrap='word',
                                         font=('Arial', 10), bg='#ffffff', relief='solid', bd=1)
        self.pest_details_text.pack(fill='both', expand=True, pady=(5, 0))

    def setup_recommendations_tab(self, parent):
        # Recommendations frame
        rec_frame = tk.Frame(parent, bg='#f0f0f0')
        rec_frame.pack(fill='both', expand=True, padx=10, pady=10)

        rec_label = tk.Label(rec_frame, text="AI Recommendations:",
                             font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='#2d5a27')
        rec_label.pack(anchor='w', pady=(0, 10))

        self.rec_text = tk.Text(rec_frame, wrap='word', font=('Arial', 11),
                                bg='#ffffff', relief='solid', bd=1, height=15)
        self.rec_text.pack(fill='both', expand=True)

        # Action buttons frame
        action_frame = tk.Frame(rec_frame, bg='#f0f0f0')
        action_frame.pack(fill='x', pady=(10, 0))

        save_btn = tk.Button(action_frame, text="Save Report",
                             command=self.save_report, bg='#2196F3', fg='white',
                             font=('Arial', 10), padx=15, pady=5)
        save_btn.pack(side='left', padx=(0, 10))

        export_btn = tk.Button(action_frame, text="Export Data",
                               command=self.export_data, bg='#4CAF50', fg='white',
                               font=('Arial', 10), padx=15, pady=5)
        export_btn.pack(side='left')

    def create_parameter_frame(self, parent, title, row):
        frame = tk.Frame(parent, bg='#f0f0f0')
        frame.pack(fill='x', padx=10, pady=5)

        label = tk.Label(frame, text=title, font=('Arial', 10), bg='#f0f0f0', width=15, anchor='w')
        label.pack(side='left')

        # Progress bar
        pb = ttk.Progressbar(frame, orient='horizontal', length=150, mode='determinate')
        pb.pack(side='left', padx=(10, 5))

        # Value label
        value_label = tk.Label(frame, text="0%", font=('Arial', 9), bg='#f0f0f0')
        value_label.pack(side='left', padx=(5, 0))

        return {'progressbar': pb, 'value_label': value_label}

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Plant Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            self.image_path = file_path
            self.load_and_display_image()

    def simulate_capture(self):
        # In a real application, this would capture from a camera
        sample_dir = "sample_plants"
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            messagebox.showinfo("Info", "Sample directory created. Please add plant images to 'sample_plants' folder.")
            return

        sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if sample_images:
            # Use a random sample image
            self.image_path = os.path.join(sample_dir, random.choice(sample_images))
            self.load_and_display_image()
        else:
            messagebox.showwarning("Warning", "No sample images found in 'sample_plants' folder.")

    def load_and_display_image(self):
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is not None:
                # Convert BGR to RGB for display
                image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image(image_rgb)

                # Reset analysis results
                self.reset_analysis()
            else:
                messagebox.showerror("Error", "Could not load the image.")

    def display_image(self, image):
        # Resize image to fit in the display area
        h, w = image.shape[:2]
        max_width, max_height = 700, 600

        # Calculate scaling factor
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h))

        # Convert to PIL Image and then to ImageTk
        pil_image = Image.fromarray(resized_image)
        tk_image = ImageTk.PhotoImage(pil_image)

        # Update image label
        self.image_label.configure(image=tk_image, text="")
        self.image_label.image = tk_image  # Keep a reference

    def analyze_plant(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload or capture an image first.")
            return

        # Simulate AI analysis for plant identification, health, soil, and pest risk
        self.simulate_comprehensive_analysis()

    def simulate_comprehensive_analysis(self):
        # Convert to HSV color space for better color analysis
        hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

        # Step 1: Identify plant species
        self.identify_plant_species(hsv_image)

        # Step 2: Analyze plant health
        self.analyze_plant_health(hsv_image)

        # Step 3: Analyze soil condition
        self.analyze_soil_condition(hsv_image)

        # Step 4: Analyze pest risk
        self.analyze_pest_risk(hsv_image)

        # Update all UI elements
        self.update_identification_ui()
        self.update_plant_health_ui()
        self.update_soil_ui()
        self.update_pest_ui()
        self.update_recommendations()

        # Create and display processed image
        self.create_processed_image(hsv_image)

    def identify_plant_species(self, hsv_image):
        # Simulate plant identification based on image characteristics
        # In a real system, this would use a trained deep learning model

        # Extract features for identification
        h, w = hsv_image.shape[:2]

        # Analyze color distribution
        green_mask = cv2.inRange(hsv_image, (35, 40, 40), (85, 255, 255))
        green_percentage = np.sum(green_mask > 0) / (h * w) * 100

        # Analyze shape characteristics (simplified)
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w) * 100

        # Simulate identification based on features
        plant_options = list(self.plant_database.keys())

        # Weighted random selection based on features
        if green_percentage > 70:
            # High green content - likely leafy plant
            weights = [0.3, 0.4, 0.1, 0.15, 0.05]  # Higher weight for roses, tomatoes
        elif green_percentage > 40:
            # Moderate green content
            weights = [0.25, 0.25, 0.2, 0.2, 0.1]
        else:
            # Low green content - likely cactus or stressed plant
            weights = [0.1, 0.1, 0.5, 0.2, 0.1]

        # Adjust based on edge density (leaf complexity)
        if edge_density > 5:
            # Complex leaf structure
            weights = [w * 1.2 for w in weights]  # Boost plants with complex leaves
            weights[2] = weights[2] * 0.5  # Reduce cactus likelihood

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Select plant based on weights
        self.identified_plant = random.choices(plant_options, weights=weights)[0]
        self.identification_confidence = random.uniform(75, 95)  # Simulated confidence

    def analyze_plant_health(self, hsv_image):
        # Calculate plant health metrics
        h, w = hsv_image.shape[:2]

        green_mask = cv2.inRange(hsv_image, (35, 40, 40), (85, 255, 255))
        green_percentage = np.sum(green_mask > 0) / (h * w) * 100

        yellow_mask = cv2.inRange(hsv_image, (20, 100, 100), (35, 255, 255))
        yellow_percentage = np.sum(yellow_mask > 0) / (h * w) * 100

        brown_mask = cv2.inRange(hsv_image, (10, 50, 20), (20, 255, 200))
        brown_percentage = np.sum(brown_mask > 0) / (h * w) * 100

        # Calculate plant health score
        base_score = green_percentage
        penalty = yellow_percentage * 0.5 + brown_percentage * 1.5
        self.plant_health_score = max(0, min(100, base_score - penalty))

        # Determine plant health status
        if self.plant_health_score >= 70:
            self.plant_health_status = "Healthy"
        elif self.plant_health_score >= 40:
            self.plant_health_status = "Moderate"
        else:
            self.plant_health_status = "Unhealthy"

    def analyze_soil_condition(self, hsv_image):
        # Simulate soil condition analysis
        # Base soil score on plant health (healthy plants usually indicate good soil)
        base_soil_score = self.plant_health_score * 0.7

        # Add some randomness to simulate variability
        soil_variation = random.uniform(-15, 15)
        self.soil_score = max(0, min(100, base_soil_score + soil_variation))

        # Determine soil condition
        if self.soil_score >= 70:
            self.soil_condition = "Good"
        elif self.soil_score >= 40:
            self.soil_condition = "Moderate"
        else:
            self.soil_condition = "Poor"

        # Simulate soil parameters
        self.nutrient_level = max(0, min(100, self.soil_score + random.uniform(-10, 10)))
        self.moisture_level = max(0, min(100, 60 + random.uniform(-20, 20)))
        self.ph_level = max(0, min(100, 50 + random.uniform(-30, 30)))

    def analyze_pest_risk(self, hsv_image):
        # Simulate pest risk analysis
        # Unhealthy plants are more susceptible to pests
        base_pest_risk = 100 - self.plant_health_score

        # Add some randomness
        pest_variation = random.uniform(-10, 10)
        self.pest_risk_score = max(0, min(100, base_pest_risk * 0.8 + pest_variation))

        # Determine pest risk level
        if self.pest_risk_score >= 70:
            self.pest_risk = "High"
        elif self.pest_risk_score >= 40:
            self.pest_risk = "Moderate"
        else:
            self.pest_risk = "Low"

        # Simulate pest detection based on identified plant
        common_pests = self.plant_database.get(self.identified_plant, {}).get("common_pests", [])

        for pest in self.pest_vars.keys():
            # Higher chance of detecting pests if risk is high and pest is common for this plant
            is_common = pest in common_pests
            detection_chance = (self.pest_risk_score / 100) * (1.5 if is_common else 0.7)

            if random.random() < detection_chance:
                self.pest_vars[pest].set("Detected")
            else:
                self.pest_vars[pest].set("Not Detected")

    def update_identification_ui(self):
        self.plant_value.config(text=self.identified_plant.title())
        self.conf_value.config(text=f"{self.identification_confidence:.1f}%")

        # Update plant information
        plant_info = self.plant_database.get(self.identified_plant, {})
        scientific_name = plant_info.get("scientific_name", "Unknown")
        family = plant_info.get("family", "Unknown")
        conditions = plant_info.get("ideal_conditions", {})
        common_pests = plant_info.get("common_pests", [])
        common_diseases = plant_info.get("common_diseases", [])

        info_text = f"""Scientific Name: {scientific_name}
Family: {family}

Ideal Growing Conditions:
- Temperature: {conditions.get('temperature', 'Unknown')}
- Sunlight: {conditions.get('sunlight', 'Unknown')}
- Water: {conditions.get('water', 'Unknown')}
- Soil pH: {conditions.get('soil_ph', 'Unknown')}

Common Pests: {', '.join(common_pests) if common_pests else 'None identified'}

Common Diseases: {', '.join(common_diseases) if common_diseases else 'None identified'}

Identification Confidence: {self.identification_confidence:.1f}%
"""

        self.plant_info_text.config(state='normal')
        self.plant_info_text.delete(1.0, tk.END)
        self.plant_info_text.insert(1.0, info_text)
        self.plant_info_text.config(state='disabled')

    def update_plant_health_ui(self):
        status_color = "#4CAF50" if self.plant_health_status == "Healthy" else "#FF9800" if self.plant_health_status == "Moderate" else "#F44336"
        self.plant_status_value.config(text=self.plant_health_status, fg=status_color)
        self.plant_score_value.config(text=f"{self.plant_health_score:.1f}%")
        self.plant_health_indicator.config(fg=status_color)

        # Update plant details
        details = f"""Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Plant Health Assessment:
- Overall Health Score: {self.plant_health_score:.1f}%
- Status: {self.plant_health_status}

Color Analysis:
- Healthy green areas: {self.calculate_green_percentage():.1f}%
- Yellow areas (potential issues): {self.calculate_yellow_percentage():.1f}%
- Brown areas (concerning): {self.calculate_brown_percentage():.1f}%

Recommendation: {self.get_plant_recommendation()}
"""
        self.plant_details_text.delete(1.0, tk.END)
        self.plant_details_text.insert(1.0, details)

    def update_soil_ui(self):
        soil_color = "#4CAF50" if self.soil_condition == "Good" else "#FF9800" if self.soil_condition == "Moderate" else "#F44336"
        self.soil_status_value.config(text=self.soil_condition, fg=soil_color)
        self.soil_score_value.config(text=f"{self.soil_score:.1f}%")

        # Update soil parameters
        self.nutrient_frame['progressbar']['value'] = self.nutrient_level
        self.nutrient_frame['value_label'].config(text=f"{self.nutrient_level:.1f}%")

        self.moisture_frame['progressbar']['value'] = self.moisture_level
        self.moisture_frame['value_label'].config(text=f"{self.moisture_level:.1f}%")

        self.ph_frame['progressbar']['value'] = self.ph_level
        self.ph_frame['value_label'].config(text=f"{self.ph_level:.1f}%")

        # Update soil details
        details = f"""Soil Condition Assessment:
- Overall Soil Quality: {self.soil_score:.1f}%
- Condition: {self.soil_condition}

Soil Parameters:
- Nutrient Levels: {self.nutrient_level:.1f}%
- Moisture Content: {self.moisture_level:.1f}%
- pH Balance: {self.ph_level:.1f}%

Recommendation: {self.get_soil_recommendation()}
"""
        self.soil_details_text.delete(1.0, tk.END)
        self.soil_details_text.insert(1.0, details)

    def update_pest_ui(self):
        pest_color = "#F44336" if self.pest_risk == "High" else "#FF9800" if self.pest_risk == "Moderate" else "#4CAF50"
        self.pest_status_value.config(text=self.pest_risk, fg=pest_color)
        self.pest_score_value.config(text=f"{self.pest_risk_score:.1f}%")

        # Update pest details
        details = f"""Pest Risk Assessment:
- Overall Pest Risk: {self.pest_risk_score:.1f}%
- Risk Level: {self.pest_risk}

Detected Issues:
"""
        for pest, var in self.pest_vars.items():
            details += f"- {pest}: {var.get()}\n"

        details += f"\nRecommendation: {self.get_pest_recommendation()}"

        self.pest_details_text.delete(1.0, tk.END)
        self.pest_details_text.insert(1.0, details)

    def update_recommendations(self):
        plant_info = self.plant_database.get(self.identified_plant, {})
        scientific_name = plant_info.get("scientific_name", "Unknown")

        recommendations = f"""AI-PLANT CARE RECOMMENDATIONS
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IDENTIFIED PLANT: {self.identified_plant.title()} ({scientific_name})
Confidence: {self.identification_confidence:.1f}%

PLANT HEALTH: {self.plant_health_status} ({self.plant_health_score:.1f}%)
{self.get_detailed_plant_recommendation()}

SOIL CONDITION: {self.soil_condition} ({self.soil_score:.1f}%)
{self.get_detailed_soil_recommendation()}

PEST RISK: {self.pest_risk} ({self.pest_risk_score:.1f}%)
{self.get_detailed_pest_recommendation()}

PLANT-SPECIFIC CARE:
{self.get_plant_specific_care()}

IMMEDIATE ACTIONS:
1. {self.get_immediate_action(1)}
2. {self.get_immediate_action(2)}
3. {self.get_immediate_action(3)}

LONG-TERM STRATEGY:
- {self.get_long_term_strategy()}

MONITORING SCHEDULE:
- Check plant health: {self.get_monitoring_schedule()}
"""
        self.rec_text.delete(1.0, tk.END)
        self.rec_text.insert(1.0, recommendations)

    def create_processed_image(self, hsv_image):
        # Create a visualization of the analysis
        processed = self.original_image.copy()

        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(processed, f"Plant: {self.identified_plant.title()}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(processed, f"Health: {self.plant_health_score:.1f}%", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(processed, f"Soil: {self.soil_score:.1f}%", (10, 90), font, 0.7, (255, 255, 255), 2)
        cv2.putText(processed, f"Pest Risk: {self.pest_risk_score:.1f}%", (10, 120), font, 0.7, (255, 255, 255), 2)

        # Convert to RGB for display
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        self.display_image(processed_rgb)

    def reset_analysis(self):
        # Reset all analysis results
        self.identified_plant = "Unknown"
        self.identification_confidence = 0
        self.plant_health_status = "Unknown"
        self.plant_health_score = 0
        self.soil_condition = "Unknown"
        self.soil_score = 0
        self.pest_risk = "Unknown"
        self.pest_risk_score = 0

        # Reset UI elements
        self.plant_value.config(text="Unknown")
        self.conf_value.config(text="0%")
        self.plant_status_value.config(text="Unknown", fg="#FF9800")
        self.plant_score_value.config(text="0%")
        self.plant_health_indicator.config(fg="#cccccc")
        self.soil_status_value.config(text="Unknown", fg="#FF9800")
        self.soil_score_value.config(text="0%")
        self.pest_status_value.config(text="Unknown", fg="#FF9800")
        self.pest_score_value.config(text="0%")

        # Reset pest detection
        for pest in self.pest_vars.keys():
            self.pest_vars[pest].set("Not Detected")

        # Clear text areas
        self.plant_info_text.config(state='normal')
        self.plant_info_text.delete(1.0, tk.END)
        self.plant_info_text.insert(1.0, "Plant information will appear here after analysis.")
        self.plant_info_text.config(state='disabled')

        self.plant_details_text.delete(1.0, tk.END)
        self.soil_details_text.delete(1.0, tk.END)
        self.pest_details_text.delete(1.0, tk.END)
        self.rec_text.delete(1.0, tk.END)

    # Helper methods for calculations (simplified)
    def calculate_green_percentage(self):
        return random.uniform(60, 90) if self.plant_health_score > 70 else random.uniform(30,
                                                                                          60) if self.plant_health_score > 40 else random.uniform(
            10, 40)

    def calculate_yellow_percentage(self):
        return random.uniform(5, 15) if self.plant_health_score > 70 else random.uniform(15,
                                                                                         35) if self.plant_health_score > 40 else random.uniform(
            30, 50)

    def calculate_brown_percentage(self):
        return random.uniform(1, 5) if self.plant_health_score > 70 else random.uniform(5,
                                                                                        15) if self.plant_health_score > 40 else random.uniform(
            20, 40)

    # Recommendation methods
    def get_plant_recommendation(self):
        if self.plant_health_score > 70:
            return "Continue current care regimen. Plant is thriving."
        elif self.plant_health_score > 40:
            return "Increase watering frequency and check sunlight exposure."
        else:
            return "Immediate attention needed. Consider soil testing and pest inspection."

    def get_soil_recommendation(self):
        if self.soil_score > 70:
            return "Soil condition is optimal. Maintain current practices."
        elif self.soil_score > 40:
            return "Consider adding organic compost and monitor drainage."
        else:
            return "Soil amendment needed. Test for specific nutrient deficiencies."

    def get_pest_recommendation(self):
        if self.pest_risk_score < 30:
            return "Low risk. Continue preventive measures."
        elif self.pest_risk_score < 60:
            return "Moderate risk. Increase monitoring and consider organic treatments."
        else:
            return "High risk. Implement integrated pest management immediately."

    def get_detailed_plant_recommendation(self):
        recommendations = {
            "Healthy": "Your plant shows excellent health. Maintain current watering and sunlight conditions. Consider light fertilization during growing season.",
            "Moderate": "Plant shows signs of stress. Check watering schedule - ensure proper drainage. Evaluate sunlight exposure and consider nutrient supplement.",
            "Unhealthy": "Immediate intervention required. Isolate plant if pests detected. Review all growing conditions and consider professional consultation."
        }
        return recommendations.get(self.plant_health_status, "Assessment in progress.")

    def get_detailed_soil_recommendation(self):
        if self.soil_score > 70:
            return "Soil composition is ideal for plant growth. Maintain organic matter content with seasonal compost addition."
        elif self.soil_score > 40:
            return "Soil requires improvement. Add well-rotted compost and consider soil testing for specific nutrient adjustments."
        else:
            return "Soil condition poor. Requires comprehensive amendment. Test pH and nutrient levels, add organic matter, and improve drainage."

    def get_detailed_pest_recommendation(self):
        if self.pest_risk_score < 30:
            return "Minimal pest concerns. Continue with preventive neem oil applications and regular monitoring."
        elif self.pest_risk_score < 60:
            return "Elevated pest risk detected. Increase inspection frequency, use insecticidal soap, and remove affected leaves."
        else:
            return "Critical pest risk. Implement immediate treatment protocol. Isolate plant, use appropriate pesticides, and consider biological controls."

    def get_plant_specific_care(self):
        plant_info = self.plant_database.get(self.identified_plant, {})
        conditions = plant_info.get("ideal_conditions", {})

        if self.identified_plant == "tomato":
            return "Tomatoes need consistent moisture and full sun. Support with stakes or cages. Watch for blossom end rot."
        elif self.identified_plant == "rose":
            return "Roses require good air circulation and regular pruning. Fertilize during growing season. Watch for black spot."
        elif self.identified_plant == "cactus":
            return "Cacti need excellent drainage and minimal water. Protect from frost. Avoid overwatering."
        elif self.identified_plant == "sunflower":
            return "Sunflowers need support in windy areas. They're heavy feeders - fertilize regularly. Watch for birds eating seeds."
        elif self.identified_plant == "orchid":
            return "Orchids need high humidity and indirect light. Water when potting mix is dry. Repot every 1-2 years."
        else:
            return f"General care: Provide appropriate sunlight ({conditions.get('sunlight', 'Unknown')}) and water ({conditions.get('water', 'Unknown')})."

    def get_immediate_action(self, priority):
        actions = {
            1: "Adjust watering schedule based on soil moisture assessment",
            2: "Apply appropriate treatment for detected issues",
            3: "Document current condition for tracking changes over time"
        }

        if self.plant_health_score < 40:
            actions = {
                1: "Immediate soil moisture correction",
                2: "Apply emergency nutrient treatment",
                3: "Isolate plant and begin pest control protocol"
            }

        return actions.get(priority, "Continue regular monitoring")

    def get_long_term_strategy(self):
        if self.plant_health_score > 70:
            return "Maintain optimal growing conditions with seasonal adjustments"
        elif self.plant_health_score > 40:
            return "Implement soil improvement plan and pest prevention schedule"
        else:
            return "Complete growing environment assessment and rehabilitation plan"

    def get_monitoring_schedule(self):
        if self.pest_risk_score > 60:
            return "Daily for next 7 days, then twice weekly"
        elif self.plant_health_score < 40:
            return "Every 2-3 days until improvement noted"
        else:
            return "Weekly assessment sufficient"

    def save_report(self):
        # Save the analysis report
        filename = f"plant_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filename, 'w') as f:
                # Get all the text from recommendations
                report_text = self.rec_text.get(1.0, tk.END)
                f.write(report_text)
            messagebox.showinfo("Success", f"Report saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save report: {str(e)}")

    def export_data(self):
        # Export analysis data
        filename = f"plant_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(filename, 'w') as f:
                f.write("Metric,Value,Unit\n")
                f.write(f"Identified Plant,{self.identified_plant},\n")
                f.write(f"Identification Confidence,{self.identification_confidence:.2f},%\n")
                f.write(f"Plant Health Score,{self.plant_health_score:.2f},%\n")
                f.write(f"Soil Quality Score,{self.soil_score:.2f},%\n")
                f.write(f"Pest Risk Score,{self.pest_risk_score:.2f},%\n")
                f.write(f"Nutrient Level,{self.nutrient_level:.2f},%\n")
                f.write(f"Moisture Content,{self.moisture_level:.2f},%\n")
                f.write(f"pH Level,{self.ph_level:.2f},%\n")
            messagebox.showinfo("Success", f"Data exported as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export data: {str(e)}")

    def run(self):
        self.root.mainloop()


# Create and run the application
if __name__ == "__main__":
    app = AdvancedPlantAnalyzer()
    app.run()