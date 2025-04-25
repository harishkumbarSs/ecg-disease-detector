# Ecg.py

from skimage.io import imread
from skimage import color, measure
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import joblib
import os


class ECG:
    def getImage(self, image_file):
        """Reads uploaded file into an RGB numpy array."""
        try:
            from PIL import Image
            img = Image.open(image_file).convert("RGB")
            return np.array(img)
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

    def GrayImage(self, image):
        """Convert to grayscale and resize to 1572×2213."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image for grayscale conversion")
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        gray = color.rgb2gray(image)
        return resize(gray, (1572, 2213), anti_aliasing=True)

    def DividingLeads(self, image):
        """Divide into 13 leads (auto‐upscale if too small), save figures, return list."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image for dividing leads")

        h, w = image.shape[:2]
        min_h, min_w = 1480, 2125
        if h < min_h or w < min_w:
            scale = max(min_h / h, min_w / w)
            image = resize(image, (int(h*scale), int(w*scale)), anti_aliasing=True)

        coords = [
            (300,600,150,643), (300,600,646,1135), (300,600,1140,1625), (300,600,1630,2125),
            (600,900,150,643), (600,900,646,1135), (600,900,1140,1625), (600,900,1630,2125),
            (900,1200,150,643),(900,1200,646,1135),(900,1200,1140,1625),(900,1200,1630,2125),
            (1250,1480,150,2125)
        ]
        leads = [image[r1:r2, c1:c2] for (r1,r2,c1,c2) in coords]

        # Save leads 1–12
        fig, axes = plt.subplots(4, 3, figsize=(10,10))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(leads[i])
            ax.set_title(f"Lead {i+1}")
            ax.axis('off')
        fig.savefig("Leads_1-12_figure.png")
        plt.close(fig)

        # Save lead 13
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.imshow(leads[12])
        ax2.set_title("Lead 13")
        ax2.axis('off')
        fig2.savefig("Long_Lead_13_figure.png")
        plt.close(fig2)

        return leads

    def PreprocessingLeads(self, leads):
        """Apply Gaussian + Otsu threshold, save two figures."""
        if not leads or len(leads) != 13:
            raise ValueError("Expected exactly 13 leads")

        # Leads 1–12
        fig, axes = plt.subplots(4, 3, figsize=(10,10))
        for i, ax in enumerate(axes.flatten()):
            y = leads[i]
            gray = color.rgb2gray(y) if y.ndim == 3 else y
            blur = gaussian(gray, sigma=1)
            t = threshold_otsu(blur)
            binary = blur < t
            small = resize(binary, (300,450), anti_aliasing=False)
            ax.imshow(small, cmap="gray")
            ax.set_title(f"Preprocessed Lead {i+1}")
            ax.axis("off")
        fig.savefig("Preprocessed_Leads_1-12_figure.png")
        plt.close(fig)

        # Lead 13
        y13 = leads[12]
        gray13 = color.rgb2gray(y13) if y13.ndim == 3 else y13
        blur13 = gaussian(gray13, sigma=1)
        t13 = threshold_otsu(blur13)
        bin13 = blur13 < t13
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.imshow(bin13, cmap='gray')
        ax3.set_title("Preprocessed Lead 13")
        ax3.axis('off')
        fig3.savefig("Preprocessed_Leads_13_figure.png")
        plt.close(fig3)

    def SignalExtraction_Scaling(self, leads):
        """
        Extract contour for leads 1–12, save contour figure,
        write 12 CSVs of length 255 (one row each, no header).
        """
        if not leads or len(leads) < 12:
            raise ValueError("Expected at least 12 leads")

        fig, axes = plt.subplots(4, 3, figsize=(10,10))
        for i, ax in enumerate(axes.flatten()):
            y = leads[i]
            gray = color.rgb2gray(y) if y.ndim == 3 else y
            blur = gaussian(gray, sigma=0.7)
            t = threshold_otsu(blur)
            binary = blur < t
            small = resize(binary, (300,450), anti_aliasing=False)
            contours = measure.find_contours(small, level=0.8)

            signal = (resize(max(contours, key=lambda c: c.shape[0]), (255,2), anti_aliasing=True)
                      if contours else np.zeros((255,2)))

            ax.plot(signal[:,1], signal[:,0])
            ax.invert_yaxis()
            ax.set_title(f"Contour Lead {i+1}")
            ax.axis("off")

            # Write exactly one data row, no header
            df = pd.DataFrame(signal[:,0]).T
            fn = f"Scaled_1DLead_{i+1}.csv"
            df.to_csv(fn, index=False, header=False)

        fig.savefig("Contour_Leads_1-12_figure.png")
        plt.close(fig)

    def CombineConvert1Dsignal(self):
        """
        Concatenate the 12 scaled CSVs into one DataFrame (12×255=3060 features),
        fill NaNs, ensure at least one row.
        """
        files = sorted(
            [f for f in os.listdir() if f.startswith("Scaled_1DLead_") and f.endswith(".csv")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        if len(files) != 12:
            raise ValueError(f"Found {len(files)} CSVs; expected 12")

        df = pd.read_csv(files[0], header=None)
        for f in files[1:]:
            df = pd.concat([df, pd.read_csv(f, header=None)], axis=1, ignore_index=True)

        df = df.fillna(0)
        if df.shape[0] == 0:
            raise ValueError("No 1D signal rows generated; cannot proceed")

        return df

    def DimensionalReduciton(self, df):
        """Apply saved PCA (expects 3060 features)."""
        pca = joblib.load("PCA_ECG (1).pkl")
        return pd.DataFrame(pca.transform(df))

    def ModelLoad_predict(self, df):
        """Load classifier and return human-readable label."""
        model = joblib.load("Heart_Disease_Prediction_using_ECG.pkl")
        pred = model.predict(df)[0]
        mapping = {
            1: "Myocardial Infarction",
            0: "Abnormal Heartbeat",
            2: "Normal",
            3: "History of Myocardial Infarction"
        }
        return f"Your ECG corresponds to {mapping.get(pred, 'Unknown')}"
