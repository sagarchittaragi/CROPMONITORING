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


class AdvancedPlantHealthAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI-Powered Plant Health, Soil & Pest Analyzer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.plant_health_status = "Unknown"
        self.plant_health_score = 0
        self.soil_condition = "Unknown"
        self.soil_score = 0
        self.pest_risk = "Unknown"
        self.pest_risk_score = 0

        self.setup_ui()

    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="AI Plant Health, Soil & Pest Risk Analyzer",
                               font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2d5a27')
        title_label.pack(pady=10)

        # Description
        desc_label = tk.Label(self.root,
                              text="Upload a plant image to analyze health, soil condition, and pest risks using AI",
                              font=('Arial', 12), bg='#f0f0f0', fg='#555555')
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
        controls_frame.pack(fill='x', pady=(0, 20))

        # Upload button
        upload_btn = tk.Button(controls_frame, text="Upload Plant Image",
                               command=self.upload_image, bg='#4CAF50', fg='white',
                               font=('Arial', 12, 'bold'), padx=20, pady=10)
        upload_btn.pack(fill='x', pady=5)

        # Capture button (simulated)
        capture_btn = tk.Button(controls_frame, text="Simulate Capture",
                                command=self.simulate_capture, bg='#2196F3', fg='white',
                                font=('Arial', 12), padx=20, pady=10)
        capture_btn.pack(fill='x', pady=5)

        # Analyze button
        analyze_btn = tk.Button(controls_frame, text="Analyze Plant Health & Risks",
                                command=self.analyze_plant, bg='#FF9800', fg='white',
                                font=('Arial', 12, 'bold'), padx=20, pady=10)
        analyze_btn.pack(fill='x', pady=5)

        # Results frame
        results_frame = tk.Frame(right_frame, bg='#f0f0f0')
        results_frame.pack(fill='both', expand=True)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill='both', expand=True, pady=10)

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

        # Soil indicator
        indicator_frame = tk.Frame(parent, bg='#f0f0f0')
        indicator_frame.pack(fill='x', padx=10, pady=10)

        self.soil_indicator = tk.Label(indicator_frame, text="⬤",
                                       font=('Arial', 24), bg='#f0f0f0', fg='#cccccc')
        self.soil_indicator.pack()

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

        # Pest indicator
        indicator_frame = tk.Frame(parent, bg='#f0f0f0')
        indicator_frame.pack(fill='x', padx=10, pady=10)

        self.pest_indicator = tk.Label(indicator_frame, text="⬤",
                                       font=('Arial', 24), bg='#f0f0f0', fg='#cccccc')
        self.pest_indicator.pack()

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
        # For simulation, we'll use a sample image
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

        # Simulate AI analysis for plant health, soil condition, and pest risk
        self.simulate_comprehensive_analysis()

    def simulate_comprehensive_analysis(self):
        # Convert to HSV color space for better color analysis
        hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

        # Extract color channels
        h, s, v = cv2.split(hsv_image)

        # Calculate plant health metrics
        green_mask = cv2.inRange(hsv_image, (35, 40, 40), (85, 255, 255))
        green_percentage = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1]) * 100

        yellow_mask = cv2.inRange(hsv_image, (20, 100, 100), (35, 255, 255))
        yellow_percentage = np.sum(yellow_mask > 0) / (yellow_mask.shape[0] * yellow_mask.shape[1]) * 100

        brown_mask = cv2.inRange(hsv_image, (10, 50, 20), (20, 255, 200))
        brown_percentage = np.sum(brown_mask > 0) / (brown_mask.shape[0] * yellow_mask.shape[1]) * 100

        # Calculate plant health score
        base_score = green_percentage
        penalty = yellow_percentage * 0.5 + brown_percentage * 1.5
        self.plant_health_score = max(0, min(100, base_score - penalty))

        # Determine plant health status
        if self.plant_health_score >= 70:
            self.plant_health_status = "Healthy"
            status_color = "#4CAF50"
        elif self.plant_health_score >= 40:
            self.plant_health_status = "Moderate"
            status_color = "#FF9800"
        else:
            self.plant_health_status = "Unhealthy"
            status_color = "#F44336"

        # Simulate soil condition analysis
        self.simulate_soil_analysis(hsv_image)

        # Simulate pest risk analysis
        self.simulate_pest_analysis(hsv_image)

        # Update all UI elements
        self.update_plant_ui(status_color)
        self.update_soil_ui()
        self.update_pest_ui()
        self.update_recommendations()

        # Create and display processed image
        self.create_processed_image(hsv_image, green_mask, yellow_mask, brown_mask)

    def simulate_soil_analysis(self, hsv_image):
        # Simulate soil condition based on plant health and image analysis
        # In a real system, this would use soil sensor data and more sophisticated analysis

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
        self.moisture_level = max(0, min(100, 60 + random.uniform(-20, 20)))  # Moisture tends to be around 60%
        self.ph_level = max(0, min(100, 50 + random.uniform(-30, 30)))  # pH balance

    def simulate_pest_analysis(self, hsv_image):
        # Simulate pest risk based on plant health and image analysis
        # Unhealthy plants are more susceptible to pests

        # Base pest risk on inverse of plant health
        base_pest_risk = 100 - self.plant_health_score

        # Add some randomness
        pest_variation = random.uniform(-10, 10)
        self.pest_risk_score = max(0, min(100, base_pest_risk * 0.8 + pest_variation))

        # Determine pest risk level
        if self.pest_risk_score >= 70:
            self.pest_risk = "High"
            risk_color = "#F44336"
        elif self.pest_risk_score >= 40:
            self.pest_risk = "Moderate"
            risk_color = "#FF9800"
        else:
            self.pest_risk = "Low"
            risk_color = "#4CAF50"

        # Simulate pest detection
        pests = ["Aphids", "Spider Mites", "Whiteflies", "Caterpillars", "Fungal Infection"]
        for pest in pests:
            # Higher chance of detecting pests if risk is high
            detection_chance = self.pest_risk_score / 100
            if random.random() < detection_chance * 0.7:  # 70% of max chance
                self.pest_vars[pest].set("Detected")
            else:
                self.pest_vars[pest].set("Not Detected")

    def update_plant_ui(self, status_color):
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
        self.soil_indicator.config(fg=soil_color)

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
        self.pest_indicator.config(fg=pest_color)

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
        recommendations = f"""AI-GROWTH RECOMMENDATIONS
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PLANT HEALTH: {self.plant_health_status} ({self.plant_health_score:.1f}%)
{self.get_detailed_plant_recommendation()}

SOIL CONDITION: {self.soil_condition} ({self.soil_score:.1f}%)
{self.get_detailed_soil_recommendation()}

PEST RISK: {self.pest_risk} ({self.pest_risk_score:.1f}%)
{self.get_detailed_pest_recommendation()}

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

    def create_processed_image(self, hsv_image, green_mask, yellow_mask, brown_mask):
        # Create a visualization of the analysis
        processed = self.original_image.copy()

        # Highlight different areas
        processed[green_mask > 0] = [0, 255, 0]  # Green for healthy
        processed[yellow_mask > 0] = [0, 255, 255]  # Yellow for moderate
        processed[brown_mask > 0] = [0, 0, 255]  # Red for concerning

        # Blend with original image
        alpha = 0.3
        processed = cv2.addWeighted(self.original_image, 1 - alpha, processed, alpha, 0)

        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(processed, f"Health: {self.plant_health_score:.1f}%", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(processed, f"Soil: {self.soil_score:.1f}%", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(processed, f"Pest Risk: {self.pest_risk_score:.1f}%", (10, 90), font, 0.7, (255, 255, 255), 2)

        # Convert to RGB for display
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        self.display_image(processed_rgb)

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
    app = AdvancedPlantHealthAnalyzer()
    app.run()