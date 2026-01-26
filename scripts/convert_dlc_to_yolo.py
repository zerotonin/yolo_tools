import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from tqdm import tqdm


class SingleFolderConverter:
    """Converts DeepLabCut annotations from a single subfolder to YOLO format"""
    
    def __init__(self, csv_file: str, subfolder_name: str, source_dir: Path, 
                 output_dir: Path, class_id: int = 0, padding_percent: float = 0.1):
        """
        Initialize converter for a single DLC subfolder
        
        Args:
            csv_file: Path to the CSV file with annotations
            subfolder_name: Name of the subfolder being processed
            source_dir: Root directory containing labeled-data
            output_dir: Output directory for YOLO format
            class_id: Class ID for the animal (default: 0)
            padding_percent: Padding around bounding box (default: 0.1 = 10%)
        """
        self.csv_file = csv_file
        self.subfolder_name = subfolder_name
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.class_id = class_id
        self.padding_percent = padding_percent
        
        self.processed = 0
        self.skipped = 0
        
    def read_csv(self) -> pd.DataFrame:
        """Read DeepLabCut CSV with multi-level headers"""
        return pd.read_csv(self.csv_file, header=[0, 1, 2])
    
    def get_bounding_box(self, row: pd.Series, img_width: int, img_height: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculate bounding box from all keypoints in a row
        
        Args:
            row: DataFrame row containing keypoint coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            Tuple of (x_center, y_center, width, height) in pixel coordinates or None
        """
        x_coords = []
        y_coords = []
        
        # Extract all x,y coordinates
        for col in row.index:
            if col[2] == 'x' and not pd.isna(row[col]):
                x_coords.append(row[col])
            elif col[2] == 'y' and not pd.isna(row[col]):
                y_coords.append(row[col])
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None
        
        # Calculate bounding box with padding
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        width = x_max - x_min
        height = y_max - y_min
        x_padding = width * self.padding_percent
        y_padding = height * self.padding_percent
        
        # Apply padding and clip to image boundaries
        x_min = max(0, x_min - x_padding)
        y_min = max(0, y_min - y_padding)
        x_max = min(img_width, x_max + x_padding)
        y_max = min(img_height, y_max + y_padding)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        return x_center, y_center, width, height
    
    @staticmethod
    def convert_to_yolo_format(x_center: float, y_center: float, 
                               width: float, height: float,
                               img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box to YOLO format (normalized coordinates)
        
        Returns:
            Tuple of (x_center_norm, y_center_norm, width_norm, height_norm)
        """
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    @staticmethod
    def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions without loading the entire image"""
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    
    def process_image(self, row: pd.Series, images_dir: Path, labels_dir: Path) -> bool:
        """
        Process a single image and create YOLO annotation
        
        Returns:
            True if successful, False otherwise
        """
        filename = row.iloc[2]
        
        # Skip if empty
        if pd.isna(filename):
            return False
        
        # Construct source image path
        source_image = self.source_dir / self.subfolder_name / filename
        
        if not source_image.exists():
            # Debug: print what we're looking for
            tqdm.write(f"  DEBUG: Looking for: {source_image}")
            tqdm.write(f"  DEBUG: Subfolder exists: {(self.source_dir / self.subfolder_name).exists()}")
            if (self.source_dir / self.subfolder_name).exists():
                files = list((self.source_dir / self.subfolder_name).glob("*"))
                tqdm.write(f"  DEBUG: Files in subfolder: {[f.name for f in files[:5]]}")
            return False
        
        # Get image dimensions first (needed for bounding box clipping)
        try:
            img_width, img_height = self.get_image_dimensions(source_image)
        except Exception as e:
            tqdm.write(f"Error reading image {source_image}: {e}")
            return False
        
        # Get bounding box
        bbox = self.get_bounding_box(row, img_width, img_height)
        
        if bbox is None:
            tqdm.write(f"Warning: No valid keypoints for {filename}")
            return False
        
        # Convert to YOLO format
        x_center, y_center, width, height = bbox
        x_norm, y_norm, w_norm, h_norm = self.convert_to_yolo_format(
            x_center, y_center, width, height, img_width, img_height
        )
        
        # Create new filename: subfolder_filename.ext
        file_stem = Path(filename).stem
        new_filename = f"{self.subfolder_name}_{filename}"
        
        # Copy image to output directory with new name
        dest_image = images_dir / new_filename
        shutil.copy2(source_image, dest_image)
        
        # Create label file with new name
        label_file = labels_dir / f"{self.subfolder_name}_{file_stem}.txt"
        with open(label_file, 'w') as f:
            f.write(f"{self.class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        return True
    
    def convert(self) -> Tuple[int, int]:
        """
        Convert all images from this subfolder
        
        Returns:
            Tuple of (processed_count, skipped_count)
        """
        # Read CSV
        df = self.read_csv()
        
        # Debug: Print first few rows to understand structure
        tqdm.write(f"\n  DEBUG: CSV has {len(df)} rows")
        tqdm.write(f"  DEBUG: First row columns [0:3]: {df.iloc[0, 0:3].tolist() if len(df) > 0 else 'empty'}")
        
        # Create output directories
        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each row with progress bar
        for idx, row in tqdm(df.iterrows(), 
                            total=len(df), 
                            desc=f"  {self.subfolder_name}",
                            leave=False):
            # Only process rows that start with 'labeled-data' (these are the actual data rows)
            # Skip header rows (scorer, bodyparts, coords)
            if row.iloc[0] != 'labeled-data':
                continue
            
            if self.process_image(row, images_dir, labels_dir):
                self.processed += 1
            else:
                self.skipped += 1
        
        return self.processed, self.skipped


class DeepLabCutConverter:
    """Converts all DeepLabCut annotations to YOLO format by processing all subfolders"""
    
    def __init__(self, labeled_data_dir: str, output_dir: str, 
                 class_id: int = 0, padding_percent: float = 0.1, create_split: bool = False):
        """
        Initialize converter for entire DLC project
        
        Args:
            labeled_data_dir: Path to the 'labeled-data' directory
            output_dir: Output directory for YOLO format dataset
            class_id: Class ID for the animal (default: 0)
            padding_percent: Padding around bounding box (default: 0.1 = 10%)
            create_split: If True, creates train/val split structure. If False, creates flat structure (default: False)
        """
        self.labeled_data_dir = Path(labeled_data_dir)
        self.output_dir = Path(output_dir)
        self.class_id = class_id
        self.padding_percent = padding_percent
        self.create_split = create_split
        
        self.total_processed = 0
        self.total_skipped = 0
    
    def find_subfolders_with_csv(self) -> list:
        """
        Find all subfolders in labeled-data that contain CSV files
        
        Returns:
            List of tuples: (subfolder_path, csv_file_path)
        """
        subfolders_with_csv = []
        
        for subfolder in self.labeled_data_dir.iterdir():
            if subfolder.is_dir():
                # Look for CSV files in this subfolder
                csv_files = list(subfolder.glob("*.csv"))
                
                if csv_files:
                    # Use the first CSV file found (typically there's only one)
                    subfolders_with_csv.append((subfolder, csv_files[0]))
        
        return subfolders_with_csv
    
    def convert(self) -> None:
        """Convert all subfolders to YOLO format"""
        print(f"Scanning for DeepLabCut subfolders in: {self.labeled_data_dir}")
        
        subfolders = self.find_subfolders_with_csv()
        
        if not subfolders:
            print("No subfolders with CSV files found!")
            return
        
        print(f"Found {len(subfolders)} subfolder(s) to process\n")
        
        # Process all subfolders with overall progress bar
        for subfolder_path, csv_file in tqdm(subfolders, 
                                             desc="Overall progress",
                                             unit="folder"):
            subfolder_name = subfolder_path.name
            
            # Create converter for this subfolder
            converter = SingleFolderConverter(
                csv_file=str(csv_file),
                subfolder_name=subfolder_name,
                source_dir=self.labeled_data_dir,
                output_dir=self.output_dir,
                class_id=self.class_id,
                padding_percent=self.padding_percent
            )
            
            # Convert
            processed, skipped = converter.convert()
            
            self.total_processed += processed
            self.total_skipped += skipped
        
        self.create_dataset_yaml()
        self.print_summary()
    
    def create_dataset_yaml(self) -> None:
        """Create YOLO dataset configuration file"""
        if self.create_split:
            # For pre-split structure
            yaml_content = f"""# YOLO Dataset Configuration
path: {self.output_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: 1  # number of classes
names: ['animal']  # class names
"""
        else:
            # For flat structure (YoloWrapper will do the split)
            yaml_content = f"""# YOLO Dataset Configuration (flat structure for YoloWrapper)
# Use YoloWrapper.create_dataset() to create train/val split

path: {self.output_dir.absolute()}
images_dir: images
labels_dir: labels

# Classes
nc: 1  # number of classes
names: ['animal']  # class names
"""
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created dataset.yaml configuration file")
    
    def print_summary(self) -> None:
        """Print conversion summary"""
        print("\n" + "="*60)
        print("CONVERSION COMPLETE!")
        print("="*60)
        print(f"Total images processed: {self.total_processed}")
        print(f"Total images skipped: {self.total_skipped}")
        print(f"\nYOLO dataset created in: {self.output_dir}")
        print(f"  - Images: {self.output_dir / 'images'}")
        print(f"  - Labels: {self.output_dir / 'labels'}")
        print(f"  - Config: {self.output_dir / 'dataset.yaml'}")
        
        if self.create_split:
            print("\nDataset structure: Pre-split (train/val folders created)")
            print("\nNext steps:")
            print("  Train with: yolo train data=dataset.yaml model=yolov8n.pt")
        else:
            print("\nDataset structure: Flat (all files in images/ and labels/)")
            print("\nNext steps:")
            print("  Use YoloWrapper.create_dataset() to create train/val split")
            print("  Or train directly with your training script")
        print("="*60)


def main():
    """Main function to run the converter"""
    
    # Configuration
    LABELED_DATA_DIR = "/projects/sciences/zoology/geurten_lab/AI_trainData/weta_temperature-Bart-2025-02-21/labeled-data"  # Your DLC labeled-data directory
    OUTPUT_DIR = "/projects/sciences/zoology/geurten_lab/AI_trainData/weta_temperature-yoloformat-data"        # Output directory for YOLO format
    CLASS_ID = 0                        # Class ID for your animal
    PADDING_PERCENT = 0.1               # 10% padding around bounding box
    CREATE_SPLIT = False                # False = flat structure for YoloWrapper, True = pre-split train/val
    
    # Create converter and run
    converter = DeepLabCutConverter(
        labeled_data_dir=LABELED_DATA_DIR,
        output_dir=OUTPUT_DIR,
        class_id=CLASS_ID,
        padding_percent=PADDING_PERCENT,
        create_split=CREATE_SPLIT
    )
    
    converter.convert()


if __name__ == "__main__":
    main()

