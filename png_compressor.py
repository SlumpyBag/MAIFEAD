# png_compressor.py (Manual Pixel Art Optimization)
"""
PNG Compressor using MAIFEAD (MAIF Encoding And Decoding)
Lossy image compression with excellent quality-to-size ratio
Drag and drop supported
Manual pixel art optimization with user-defined pixel size
"""

import numpy as np
from PIL import Image
import colorsys
import struct
import zlib
import os
import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Tuple, Dict
import threading
import time
from datetime import datetime

# Try to import tkinterdnd2 for drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

class MAIFEADCompressor:
    """MAIF Encoding And Decoding - Combined compression engine"""
    
    def __init__(self, quality: str = "high", preserve_edges: bool = True, 
                 pixel_art_pixel_size: int = 0):
        """
        Initialize MAIFEAD Compressor
        quality: "excellent", "high", "balanced", "compact"
        preserve_edges: preserve high-frequency edge details
        pixel_art_pixel_size: size of pixels in pixel art (0 = disabled)
        """
        self.quality = quality
        self.preserve_edges = preserve_edges
        self.pixel_art_pixel_size = pixel_art_pixel_size  # 0 = disabled
        
        # Quality presets (threshold values)
        self.quality_presets = {
            "excellent": 0.0003,
            "high": 0.0008,
            "balanced": 0.0015,
            "compact": 0.004
        }
        
        self.quality_threshold = self.quality_presets.get(quality, 0.001)
        self.width = 0
        self.height = 0
        self.channels = {}
        self.metadata = {}
        self.compression_stats = {}
        self.actual_scale = 1  # Actual scale factor used
        
    def scale_image_for_pixel_art(self, image: Image.Image, pixel_size: int) -> Tuple[Image.Image, int]:
        """
        Scale image down based on pixel size
        pixel_size: size of each pixel in the image (e.g., 8 for 8x8 pixel art)
        Returns (scaled_image, scale_factor_used)
        """
        if pixel_size <= 1:
            return image, 1
        
        width, height = image.size
        
        # Calculate how many pixels wide/tall the image is
        pixels_wide = width // pixel_size
        pixels_tall = height // pixel_size
        
        # If the image doesn't divide evenly, use the nearest division
        if width % pixel_size != 0 or height % pixel_size != 0:
            # Adjust pixel size to divide evenly
            for scale in range(pixel_size, 1, -1):
                if width % scale == 0 and height % scale == 0:
                    pixel_size = scale
                    pixels_wide = width // pixel_size
                    pixels_tall = height // pixel_size
                    break
        
        # If we can't divide evenly, use the closest fit
        if width % pixel_size != 0 or height % pixel_size != 0:
            # Just use the original pixel size and let resize handle it
            pass
        
        # Scale down using nearest neighbor
        scaled = image.resize((pixels_wide, pixels_tall), Image.Resampling.NEAREST)
        
        # Calculate actual scale factor
        scale_factor = width // pixels_wide
        
        return scaled, scale_factor
    
    def upscale_pixel_art(self, image: Image.Image, scale_factor: int, 
                          original_width: int, original_height: int) -> Image.Image:
        """Upscale pixel art back to original size using nearest neighbor"""
        if scale_factor <= 1:
            return image
        
        # Use nearest neighbor to preserve pixel art look
        return image.resize((original_width, original_height), Image.Resampling.NEAREST)
    
    def rgb_to_hsl(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSL - vectorized"""
        rgb_normalized = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_normalized[:, :, 0], rgb_normalized[:, :, 1], rgb_normalized[:, :, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        l = (max_val + min_val) / 2
        
        s = np.zeros_like(l)
        mask = diff != 0
        s[mask] = diff[mask] / (1 - np.abs(2 * l[mask] - 1))
        
        h = np.zeros_like(l)
        mask = (max_val == r) & (diff != 0)
        h[mask] = ((g[mask] - b[mask]) / diff[mask]) % 6
        mask = (max_val == g) & (diff != 0)
        h[mask] = 2 + (b[mask] - r[mask]) / diff[mask]
        mask = (max_val == b) & (diff != 0)
        h[mask] = 4 + (r[mask] - g[mask]) / diff[mask]
        h = (h / 6.0) % 1.0
        
        return np.stack([h, s, l], axis=2)
    
    def hsl_to_rgb(self, hsl: np.ndarray) -> np.ndarray:
        """Convert HSL to RGB - vectorized"""
        h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]
        
        c = (1 - np.abs(2 * l - 1)) * s
        x = c * (1 - np.abs((h * 6) % 2 - 1))
        m = l - c / 2
        
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        sector = (h * 6).astype(np.int32) % 6
        
        masks = [
            sector == 0,
            sector == 1,
            sector == 2,
            sector == 3,
            sector == 4,
            sector == 5
        ]
        
        r[masks[0]] = c[masks[0]]
        g[masks[0]] = x[masks[0]]
        b[masks[0]] = 0
        
        r[masks[1]] = x[masks[1]]
        g[masks[1]] = c[masks[1]]
        b[masks[1]] = 0
        
        r[masks[2]] = 0
        g[masks[2]] = c[masks[2]]
        b[masks[2]] = x[masks[2]]
        
        r[masks[3]] = 0
        g[masks[3]] = x[masks[3]]
        b[masks[3]] = c[masks[3]]
        
        r[masks[4]] = x[masks[4]]
        g[masks[4]] = 0
        b[masks[4]] = c[masks[4]]
        
        r[masks[5]] = c[masks[5]]
        g[masks[5]] = 0
        b[masks[5]] = x[masks[5]]
        
        rgb = np.stack([r + m, g + m, b + m], axis=2)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    def apply_fourier_transform(self, channel: np.ndarray) -> np.ndarray:
        """Apply 2D Fourier transform"""
        return np.fft.fft2(channel)
    
    def inverse_fourier_transform(self, freq_data: np.ndarray) -> np.ndarray:
        """Apply inverse 2D Fourier transform"""
        return np.fft.ifft2(freq_data).real
    
    def reduce_frequencies_adaptive(self, freq_data: np.ndarray, is_pixel_art: bool = False) -> Tuple[np.ndarray, Dict]:
        """Adaptively reduce frequencies while preserving important details"""
        amplitudes = np.abs(freq_data)
        max_amp = np.max(amplitudes)
        
        # For pixel art, use a much more aggressive reduction since we'll upscale later
        if is_pixel_art:
            threshold = max_amp * (self.quality_threshold * 0.4)  # Even more aggressive for pixel art
        else:
            threshold = max_amp * self.quality_threshold
        
        h, w = freq_data.shape
        center_h, center_w = h // 2, w // 2
        
        # Distance-based weighting for edge preservation
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_distance = np.sqrt(center_h**2 + center_w**2)
        normalized_distance = distance / max_distance if max_distance > 0 else np.zeros_like(distance)
        
        if self.preserve_edges:
            # Adaptive threshold: preserve high frequencies for edges
            adaptive_threshold = threshold * (0.3 + 0.7 * (1 - normalized_distance))
        else:
            adaptive_threshold = threshold * np.ones_like(normalized_distance)
        
        mask = amplitudes >= adaptive_threshold
        
        # For pixel art, keep fewer frequencies (we'll upscale later)
        if is_pixel_art:
            min_freq_to_keep = int(freq_data.size * 0.05)  # Only 5% for pixel art
        else:
            min_freq_to_keep = int(freq_data.size * 0.15)  # 15% for normal images
        
        if np.sum(mask) < min_freq_to_keep:
            flat_amplitudes = amplitudes.flatten()
            indices = np.argsort(flat_amplitudes)[-min_freq_to_keep:]
            mask_flat = np.zeros_like(flat_amplitudes, dtype=bool)
            mask_flat[indices] = True
            mask = mask_flat.reshape(h, w)
        
        reduced_freq = freq_data * mask
        
        metadata = {
            'kept_frequencies': int(np.sum(mask)),
            'total_frequencies': freq_data.size,
            'threshold': self.quality_threshold,
            'preserve_edges': self.preserve_edges,
            'is_pixel_art': is_pixel_art
        }
        
        return reduced_freq, metadata
    
    def compress(self, image_path: str, progress_callback=None) -> Tuple[bytes, Dict]:
        """Compress an image using MAIFEAD"""
        # Load image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_width, original_height = img.size
        is_pixel_art = False
        scale_factor = 1
        scaled_width = original_width
        scaled_height = original_height
        
        # Check if pixel art optimization is enabled
        if self.pixel_art_pixel_size > 1:
            if progress_callback:
                progress_callback(5, f"Optimizing for pixel art (pixel size: {self.pixel_art_pixel_size})...")
            
            is_pixel_art = True
            img, scale_factor = self.scale_image_for_pixel_art(img, self.pixel_art_pixel_size)
            scaled_width, scaled_height = img.size
            
            if progress_callback:
                progress_callback(8, f"Scaled from {original_width}x{original_height} to {scaled_width}x{scaled_height} ({scale_factor}x)")
        
        rgb_array = np.array(img)
        height, width, _ = rgb_array.shape
        self.width = width
        self.height = height
        self.actual_scale = scale_factor
        
        if progress_callback:
            progress_callback(10, f"Converting {width}x{height} image to HSL...")
        
        # Convert to HSL
        hsl_array = self.rgb_to_hsl(rgb_array)
        
        # Process each channel with different thresholds
        channels = {}
        channel_metadata = {}
        
        # Different thresholds for different channels
        channel_thresholds = {
            'H': self.quality_threshold * 1.8,  # More compression for hue
            'S': self.quality_threshold * 0.7,  # Less compression for saturation
            'L': self.quality_threshold * 0.5   # Least compression for lightness
        }
        
        for channel_idx, channel_name in enumerate(['H', 'S', 'L']):
            if progress_callback:
                progress_callback(20 + channel_idx * 25, f"Processing {channel_name} channel...")
            
            channel_data = hsl_array[:, :, channel_idx]
            freq_data = self.apply_fourier_transform(channel_data)
            
            original_threshold = self.quality_threshold
            self.quality_threshold = channel_thresholds[channel_name]
            
            reduced_freq, metadata = self.reduce_frequencies_adaptive(freq_data, is_pixel_art)
            channel_metadata[channel_name] = metadata
            
            self.quality_threshold = original_threshold
            
            real_parts = reduced_freq.real.astype(np.float32)
            imag_parts = reduced_freq.imag.astype(np.float32)
            
            channels[channel_name] = {
                'real': real_parts,
                'imag': imag_parts
            }
        
        # Prepare compression data
        maif_data = {
            'width': width,
            'height': height,
            'original_width': original_width,
            'original_height': original_height,
            'scaled_width': scaled_width,
            'scaled_height': scaled_height,
            'channels': channels,
            'metadata': channel_metadata,
            'format_version': '1.0',
            'quality_settings': {
                'quality': self.quality,
                'threshold': self.quality_threshold,
                'preserve_edges': self.preserve_edges,
                'is_pixel_art': is_pixel_art,
                'pixel_art_scale': scale_factor if is_pixel_art else 1,
                'pixel_art_pixel_size': self.pixel_art_pixel_size
            }
        }
        
        if progress_callback:
            progress_callback(95, "Serializing and compressing...")
        
        # Serialize
        compressed_data = self.serialize_maif(maif_data)
        
        # Calculate stats
        original_size = os.path.getsize(image_path)
        compressed_size = len(compressed_data)
        
        self.compression_stats = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': compressed_size / original_size if original_size > 0 else 0,
            'savings': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
            'dimensions': f"{original_width}x{original_height}",
            'scaled_dimensions': f"{scaled_width}x{scaled_height}" if is_pixel_art else None,
            'quality': self.quality,
            'is_pixel_art': is_pixel_art,
            'pixel_art_scale': scale_factor if is_pixel_art else 1,
            'pixel_art_pixel_size': self.pixel_art_pixel_size if is_pixel_art else 0
        }
        
        return compressed_data, self.compression_stats
    
    def decompress(self, compressed_data: bytes, progress_callback=None) -> Image.Image:
        """Decompress MAIFEAD data back to image"""
        if progress_callback:
            progress_callback(10, "Decompressing data...")
        
        # Deserialize
        maif_dict = self.deserialize_maif(compressed_data)
        
        if progress_callback:
            progress_callback(30, "Reconstructing channels...")
        
        # Reconstruct HSL channels
        hsl_array = np.zeros((self.height, self.width, 3), dtype=np.float32)
        is_pixel_art = maif_dict['quality_settings'].get('is_pixel_art', False)
        
        for idx, channel_name in enumerate(['H', 'S', 'L']):
            if progress_callback:
                progress_callback(40 + idx * 18, f"Reconstructing {channel_name} channel...")
            
            channel = maif_dict['channels'][channel_name]
            complex_data = channel['real'] + 1j * channel['imag']
            reconstructed = self.inverse_fourier_transform(complex_data)
            
            if channel_name == 'H':
                reconstructed = np.clip(reconstructed, 0, 1)
            elif channel_name == 'S':
                reconstructed = np.clip(reconstructed, 0, 1)
                if not is_pixel_art:
                    reconstructed = np.clip(reconstructed * 1.02, 0, 1)  # Slight saturation boost
            elif channel_name == 'L':
                reconstructed = np.clip(reconstructed, 0, 1)
                if not is_pixel_art:
                    reconstructed = np.clip((reconstructed - 0.5) * 1.02 + 0.5, 0, 1)  # Slight contrast boost
                
            hsl_array[:, :, idx] = reconstructed
        
        if progress_callback:
            progress_callback(95, "Converting to RGB...")
        
        rgb_array = self.hsl_to_rgb(hsl_array)
        img = Image.fromarray(rgb_array, 'RGB')
        
        # If pixel art, upscale back to original size
        if is_pixel_art and maif_dict['quality_settings'].get('pixel_art_scale', 1) > 1:
            scale = maif_dict['quality_settings']['pixel_art_scale']
            original_width = maif_dict.get('original_width', self.width * scale)
            original_height = maif_dict.get('original_height', self.height * scale)
            
            if progress_callback:
                progress_callback(97, f"Upscaling pixel art back to {original_width}x{original_height}...")
            
            img = img.resize((original_width, original_height), Image.Resampling.NEAREST)
        
        if progress_callback:
            progress_callback(100, "Done!")
        
        return img
    
    def serialize_maif(self, maif_data: Dict) -> bytes:
        """Serialize MAIF data to binary format"""
        header = struct.pack('!II', maif_data['width'], maif_data['height'])
        
        # Store quality settings
        quality_json = json.dumps(maif_data['quality_settings']).encode()
        quality_header = struct.pack('!I', len(quality_json)) + quality_json + b'\x00'
        
        channel_data = b''
        for channel_name in ['H', 'S', 'L']:
            channel = maif_data['channels'][channel_name]
            real_bytes = channel['real'].tobytes()
            imag_bytes = channel['imag'].tobytes()
            
            meta = maif_data['metadata'][channel_name]
            meta_bytes = json.dumps({
                'kept_frequencies': meta['kept_frequencies'],
                'total_frequencies': meta['total_frequencies'],
                'threshold': meta['threshold']
            }).encode()
            
            channel_header = struct.pack('!I', len(real_bytes)) + meta_bytes + b'\x00'
            channel_data += channel_header + real_bytes + imag_bytes
        
        full_data = quality_header + header + channel_data
        
        # Use balanced compression
        return zlib.compress(full_data, level=6)
    
    def deserialize_maif(self, maif_data: bytes) -> Dict:
        """Deserialize MAIF binary data"""
        decompressed_data = zlib.decompress(maif_data)
        offset = 0
        
        # Read quality settings
        quality_settings = {}
        if offset + 4 <= len(decompressed_data):
            quality_size = struct.unpack('!I', decompressed_data[offset:offset+4])[0]
            offset += 4
            
            quality_end = decompressed_data.find(b'\x00', offset)
            if quality_end != -1:
                quality_bytes = decompressed_data[offset:quality_end]
                try:
                    quality_settings = json.loads(quality_bytes.decode())
                    self.quality = quality_settings.get('quality', 'high')
                    self.quality_threshold = quality_settings.get('threshold', 0.001)
                    self.preserve_edges = quality_settings.get('preserve_edges', True)
                    self.pixel_art_pixel_size = quality_settings.get('pixel_art_pixel_size', 0)
                except:
                    pass
                offset = quality_end + 1
        
        # Read header
        width, height = struct.unpack('!II', decompressed_data[offset:offset+8])
        offset += 8
        self.width = width
        self.height = height
        
        # Parse channels
        channels = {}
        metadata = {}
        
        for channel_name in ['H', 'S', 'L']:
            if offset + 4 > len(decompressed_data):
                break
                
            real_size = struct.unpack('!I', decompressed_data[offset:offset+4])[0]
            offset += 4
            
            meta_end = decompressed_data.find(b'\x00', offset)
            if meta_end == -1:
                break
                
            meta_bytes = decompressed_data[offset:meta_end]
            meta = json.loads(meta_bytes.decode())
            metadata[channel_name] = meta
            offset = meta_end + 1
            
            real_bytes = decompressed_data[offset:offset+real_size]
            offset += real_size
            
            imag_size = real_size
            imag_bytes = decompressed_data[offset:offset+imag_size]
            offset += imag_size
            
            real_data = np.frombuffer(real_bytes, dtype=np.float32).reshape(height, width)
            imag_data = np.frombuffer(imag_bytes, dtype=np.float32).reshape(height, width)
            
            channels[channel_name] = {
                'real': real_data,
                'imag': imag_data
            }
        
        self.channels = channels
        self.metadata = metadata
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'metadata': metadata,
            'quality_settings': quality_settings,
            'original_width': quality_settings.get('original_width', width),
            'original_height': quality_settings.get('original_height', height)
        }

class CompressorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MAIFEAD Compressor - PNG to MAIF to PNG")
        self.root.geometry("700x580")
        self.root.configure(bg='#f0f0f0')
        
        # Setup drag and drop if available
        if HAS_DND:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        
        self.compressor = None
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.browse_files())
        self.root.bind('<Return>', lambda e: self.compress_files())
        
    def setup_ui(self):
        # Title with branding
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10)
        
        title = tk.Label(title_frame, text="MAIFEAD Compressor", 
                        font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title.pack()
        
        subtitle = tk.Label(title_frame, 
                           text="MAIF Encoding And Decoding - Advanced Lossy Compression",
                           font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d')
        subtitle.pack()
        
        # Drop zone
        self.drop_frame = tk.Frame(self.root, bg='white', relief=tk.RAISED, bd=3)
        self.drop_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        drop_text = "Drop PNG/JPG Images Here\nor click to browse\n\nMAIFEAD will compress and decompress automatically"
        if not HAS_DND:
            drop_text = "Click to browse for images\n\nMAIFEAD will compress and decompress automatically"
        
        self.drop_label = tk.Label(self.drop_frame, 
                                  text=drop_text,
                                  font=('Arial', 14), bg='white', fg='#7f8c8d')
        self.drop_label.pack(expand=True)
        
        self.drop_frame.bind('<Button-1>', self.browse_files)
        self.drop_label.bind('<Button-1>', self.browse_files)
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Quality settings
        quality_frame = tk.Frame(control_frame, bg='#f0f0f0')
        quality_frame.pack(side=tk.LEFT, expand=True)
        
        tk.Label(quality_frame, text="Compression Quality:", 
                font=('Arial', 10, 'bold'), bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        
        self.quality_var = tk.StringVar(value="High")
        quality_presets = ["Excellent", "High", "Balanced", "Compact"]
        quality_menu = ttk.Combobox(quality_frame, textvariable=self.quality_var, 
                                   values=quality_presets, width=10, state='readonly')
        quality_menu.pack(side=tk.LEFT, padx=5)
        
        # Edge preservation
        self.edge_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Preserve Edges", 
                      variable=self.edge_var, bg='#f0f0f0',
                      font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        
        # Pixel art optimization frame (new row)
        pixel_frame = tk.Frame(self.root, bg='#f0f0f0')
        pixel_frame.pack(pady=5, fill=tk.X, padx=20)
        
        # Pixel art checkbox
        self.pixel_art_var = tk.BooleanVar(value=False)
        tk.Checkbutton(pixel_frame, text="Optimize for Pixel Art", 
                      variable=self.pixel_art_var, bg='#f0f0f0',
                      font=('Arial', 10, 'bold'),
                      command=self.toggle_pixel_art).pack(side=tk.LEFT, padx=5)
        
        # Pixel size entry
        tk.Label(pixel_frame, text="Pixel Size (e.g., 8, 16, 32):", 
                bg='#f0f0f0', font=('Arial', 9)).pack(side=tk.LEFT, padx=10)
        
        self.pixel_size_var = tk.StringVar(value="8")
        self.pixel_size_entry = tk.Entry(pixel_frame, textvariable=self.pixel_size_var, 
                                        width=8, state='disabled')
        self.pixel_size_entry.pack(side=tk.LEFT, padx=5)
        
        # Help text
        tk.Label(pixel_frame, text="(Enter the size of each pixel in your image)", 
                bg='#f0f0f0', font=('Arial', 8), fg='#7f8c8d').pack(side=tk.LEFT, padx=5)
        
        # Compress button
        self.compress_btn = tk.Button(self.root, text="Compress Selected", 
                                     command=self.compress_files,
                                     bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                                     padx=20, pady=5)
        self.compress_btn.pack(pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, 
                                           maximum=100, length=600)
        self.progress_bar.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready - Drop images or click to browse", 
                                    bg='#f0f0f0', font=('Arial', 9))
        self.status_label.pack()
        
        # Results display
        self.results_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.results_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.results_text = tk.Text(self.results_frame, height=5, width=70, 
                                   font=('Courier', 9), bg='white')
        self.results_text.pack(fill=tk.X)
        
        # Stats
        self.stats_label = tk.Label(self.root, text="", bg='#f0f0f0', font=('Arial', 9))
        self.stats_label.pack()
        
        # Selected files
        self.files = []
        self.file_list = tk.Listbox(self.root, height=3, font=('Arial', 9))
        self.file_list.pack(pady=5, fill=tk.X, padx=20)
        self.file_list.bind('<Double-Button-1>', lambda e: self.compress_files())
        
        # Clear button
        clear_btn = tk.Button(self.root, text="Clear List", 
                             command=self.clear_files,
                             bg='#e74c3c', fg='white', font=('Arial', 9),
                             padx=10, pady=2)
        clear_btn.pack(pady=2)
        
        # Info
        info = tk.Label(self.root, 
                       text="Tip: Pixel Art optimization scales down by pixel size, compresses, then scales back up",
                       font=('Arial', 8), bg='#f0f0f0', fg='#95a5a6')
        info.pack(pady=2)
    
    def toggle_pixel_art(self):
        """Enable/disable pixel size entry based on checkbox"""
        if self.pixel_art_var.get():
            self.pixel_size_entry.config(state='normal')
        else:
            self.pixel_size_entry.config(state='disabled')
    
    def clear_files(self):
        """Clear the file list"""
        self.files = []
        self.file_list.delete(0, tk.END)
        self.status_label.config(text="File list cleared")
        self.results_text.delete(1.0, tk.END)
        self.stats_label.config(text="")
    
    def on_drop(self, event):
        """Handle dropped files"""
        if HAS_DND:
            files = self.root.tk.splitlist(event.data)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    if file not in self.files:
                        self.files.append(file)
                        self.file_list.insert(tk.END, os.path.basename(file))
                else:
                    messagebox.showwarning("Warning", f"Skipping unsupported file: {os.path.basename(file)}")
            
            if self.files:
                self.status_label.config(text=f"Loaded {len(self.files)} files. Click 'Compress' to process.")
    
    def browse_files(self, event=None):
        """Open file browser"""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), 
                      ("All files", "*.*")]
        )
        for file in files:
            if file not in self.files:
                self.files.append(file)
                self.file_list.insert(tk.END, os.path.basename(file))
        
        if self.files:
            self.status_label.config(text=f"Loaded {len(self.files)} files. Click 'Compress' to process.")
    
    def compress_files(self):
        """Compress all selected files"""
        if not self.files:
            messagebox.showinfo("Info", "Please select some images first")
            return
        
        # Get pixel size if enabled
        pixel_size = 0
        if self.pixel_art_var.get():
            try:
                pixel_size = int(self.pixel_size_var.get())
                if pixel_size < 2:
                    messagebox.showerror("Error", "Pixel size must be at least 2")
                    return
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid pixel size (number)")
                return
        
        self.compress_btn.config(state='disabled')
        self.status_label.config(text="Compressing...")
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)
        self.stats_label.config(text="")
        
        def compression_thread():
            total_files = len(self.files)
            results = []
            successful_compressions = 0
            
            for idx, input_path in enumerate(self.files):
                try:
                    # Update progress
                    progress_base = (idx / total_files) * 100
                    self.progress_var.set(progress_base)
                    self.status_label.config(text=f"Processing {idx+1}/{total_files}: {os.path.basename(input_path)}")
                    self.root.update()
                    
                    # Get compression settings
                    quality = self.quality_var.get().lower()
                    preserve_edges = self.edge_var.get()
                    
                    # Compress with pixel art optimization if enabled
                    compressor = MAIFEADCompressor(
                        quality=quality, 
                        preserve_edges=preserve_edges,
                        pixel_art_pixel_size=pixel_size
                    )
                    
                    def update_progress(value, message):
                        relative_progress = progress_base + (value / total_files)
                        self.progress_var.set(min(relative_progress, 100))
                        self.status_label.config(text=message)
                        self.root.update()
                    
                    # Step 1: Compress (encode)
                    compressed_data, stats = compressor.compress(input_path, update_progress)
                    
                    # Step 2: Decompress (decode) directly in memory
                    update_progress(50, "Decompressing back to PNG...")
                    img = compressor.decompress(compressed_data, update_progress)
                    
                    # Step 3: Save
                    output_path = os.path.splitext(input_path)[0] + '_compressed.png'
                    img.save(output_path, 'PNG', optimize=True)
                    
                    # Step 4: Get final sizes
                    final_size = os.path.getsize(output_path)
                    
                    result = {
                        'input': os.path.basename(input_path),
                        'output': os.path.basename(output_path),
                        'original_size': stats['original_size'],
                        'compressed_size': stats['compressed_size'],
                        'final_size': final_size,
                        'ratio': stats['ratio'],
                        'savings': stats['savings'],
                        'quality': quality,
                        'is_pixel_art': stats.get('is_pixel_art', False),
                        'scale': stats.get('pixel_art_scale', 1),
                        'pixel_size': stats.get('pixel_art_pixel_size', 0)
                    }
                    results.append(result)
                    successful_compressions += 1
                    
                    # Update results display
                    self.display_results(results)
                    
                except Exception as e:
                    results.append({
                        'input': os.path.basename(input_path),
                        'error': str(e)
                    })
                    self.display_results(results)
            
            # Final status
            self.progress_var.set(100)
            
            if successful_compressions > 0:
                total_savings = sum([r.get('savings', 0) for r in results if 'savings' in r])
                avg_savings = total_savings / successful_compressions if successful_compressions > 0 else 0
                
                pixel_art_count = sum([1 for r in results if r.get('is_pixel_art', False)])
                
                status_msg = f"Done! Processed {successful_compressions}/{total_files} files"
                if pixel_art_count > 0:
                    status_msg += f" (Pixel art optimized: {pixel_art_count})"
                self.status_label.config(text=status_msg)
                self.stats_label.config(text=f"Average size reduction: {avg_savings:.1f}%")
                
                messagebox.showinfo("Compression Complete", 
                    f"Processed {successful_compressions}/{total_files} files successfully\n"
                    f"Average size reduction: {avg_savings:.1f}%\n"
                    f"Pixel art optimized: {pixel_art_count}\n"
                    f"Check *_compressed.png files in original folders")
            else:
                self.status_label.config(text="No files were compressed successfully")
                messagebox.showerror("Error", "All files failed to compress. Check the results for details.")
            
            self.compress_btn.config(state='normal')
        
        thread = threading.Thread(target=compression_thread)
        thread.start()
    
    def display_results(self, results):
        """Display compression results"""
        self.results_text.delete(1.0, tk.END)
        
        for result in results:
            if 'error' in result:
                self.results_text.insert(tk.END, f"❌ {result['input']}: {result['error']}\n")
            else:
                original_mb = result['original_size'] / (1024 * 1024)
                final_mb = result['final_size'] / (1024 * 1024)
                savings = result['savings']
                
                pixel_art_tag = " 🎨 Pixel Art" if result.get('is_pixel_art', False) else ""
                scale_info = f" (scaled {result.get('scale', 1)}x, pixel size {result.get('pixel_size', 0)})" if result.get('is_pixel_art', False) else ""
                
                line = (f"✅ {result['input']}: {original_mb:.2f}MB → {final_mb:.2f}MB "
                       f"({savings:.1f}% reduction) - {result['quality']}{pixel_art_tag}{scale_info}\n")
                self.results_text.insert(tk.END, line)
        
        self.results_text.see(tk.END)
        self.root.update()

def main():
    # Command line mode
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        print("MAIFEAD Compressor - Command Line Mode")
        print("=" * 50)
        print()
        
        successful = 0
        for input_path in files:
            if not os.path.exists(input_path):
                print(f"❌ File not found: {input_path}")
                continue
            
            if not input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                print(f"❌ Unsupported format: {input_path}")
                continue
            
            try:
                print(f"\n📁 Processing: {os.path.basename(input_path)}")
                print("Compressing...")
                
                # Use high quality without pixel art optimization by default
                compressor = MAIFEADCompressor(quality="high", preserve_edges=True, pixel_art_pixel_size=0)
                
                def progress(value, message):
                    print(f"  {value}%: {message}")
                
                # Compress
                compressed_data, stats = compressor.compress(input_path, progress)
                
                # Decompress
                print("Decompressing...")
                img = compressor.decompress(compressed_data, progress)
                
                # Save
                output_path = os.path.splitext(input_path)[0] + '_compressed.png'
                img.save(output_path, 'PNG', optimize=True)
                
                # Results
                original_mb = stats['original_size'] / (1024 * 1024)
                final_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                print(f"\n✅ Success!")
                print(f"   Original: {original_mb:.2f} MB")
                print(f"   Compressed: {final_mb:.2f} MB")
                print(f"   Reduction: {stats['savings']:.1f}%")
                if stats.get('is_pixel_art', False):
                    print(f"   🎨 Pixel art optimized (scaled {stats.get('pixel_art_scale', 1)}x)")
                print(f"   Saved to: {output_path}")
                successful += 1
                
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(input_path)}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*50}")
        print(f"Processed {successful}/{len(files)} files successfully")
        input("\nPress Enter to exit...")
    
    else:
        # GUI mode
        if HAS_DND:
            root = TkinterDnD.Tk()
        else:
            root = tk.Tk()
        app = CompressorGUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()