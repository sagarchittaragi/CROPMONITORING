import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt


class MultispectralPlantAnalyzer:
    def __init__(self):
        self.spectral_bands = {
            'visible': (400, 700),
            'nir': (700, 1100),
            'red_edge': (680, 730)
        }

    def simulate_multispectral_analysis(self, image_path):
        """Simulate multispectral analysis from RGB image"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Simulate different spectral bands
        ndvi = self.calculate_ndvi(img_rgb)
        gndvi = self.calculate_gndvi(img_rgb)
        ndwi = self.calculate_ndwi(img_rgb)

        return {
            'ndvi': ndvi,
            'gndvi': gndvi,
            'ndwi': ndwi,
            'health_index': self.calculate_health_index(ndvi, gndvi, ndwi)
        }

    def calculate_ndvi(self, img_rgb):
        """Calculate Normalized Difference Vegetation Index"""
        # Simulate NIR and Red bands
        nir = img_rgb[:, :, 0].astype(float)  # Using red channel as proxy for NIR
        red = img_rgb[:, :, 2].astype(float)  # Using blue channel as proxy for Red

        ndvi = (nir - red) / (nir + red + 1e-8)
        return np.mean(ndvi)

    def calculate_gndvi(self, img_rgb):
        """Calculate Green Normalized Difference Vegetation Index"""
        nir = img_rgb[:, :, 0].astype(float)
        green = img_rgb[:, :, 1].astype(float)

        gndvi = (nir - green) / (nir + green + 1e-8)
        return np.mean(gndvi)

    def calculate_ndwi(self, img_rgb):
        """Calculate Normalized Difference Water Index"""
        green = img_rgb[:, :, 1].astype(float)
        nir = img_rgb[:, :, 0].astype(float)

        ndwi = (green - nir) / (green + nir + 1e-8)
        return np.mean(ndwi)

    def calculate_health_index(self, ndvi, gndvi, ndwi):
        """Calculate overall plant health index"""
        return (ndvi * 0.5 + gndvi * 0.3 + ndwi * 0.2) * 100


# Add this to the main PlantHealthAnalyzer class
def enhanced_analysis(self, image_path):
    """Enhanced analysis with multispectral simulation"""
    basic_results = self.analyze_plant_health(image_path)

    # Add multispectral analysis
    spectral_analyzer = MultispectralPlantAnalyzer()
    spectral_results = spectral_analyzer.simulate_multispectral_analysis(image_path)

    basic_results['spectral_analysis'] = spectral_results
    basic_results['health_score'] = (basic_results['health_score'] + spectral_results['health_index']) / 2

    return basic_results
if __name__ == "__main__":
    app = AdvancedPlantAnalyzer()
    app.run()
