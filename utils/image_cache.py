"""
Image Cache Module
Caches images in RAM to avoid repeated disk I/O during training.

Usage:
    from utils.image_cache import get_global_cache
    
    cache = get_global_cache(enable=True, max_size_mb=2048)
    normalized_img = cache.get_or_compute(image_path, normalize_fn)
"""

from collections import OrderedDict
from PIL import Image
import threading


class ImageCache:
    """
    Thread-safe LRU cache for images in RAM.
    
    Caches images to avoid:
    - Repeated disk I/O
    - Repeated image loading/processing
    
    Args:
        enable: Enable caching (default: True)
        max_size_mb: Maximum cache size in MB (default: 2048 = 2GB)
        verbose: Print cache statistics (default: False)
    """
    
    def __init__(self, enable=True, max_size_mb=2048, verbose=False):
        self.enable = enable
        self.max_size_mb = max_size_mb
        self.verbose = verbose
        self._lock = threading.Lock()
        
        # Thread-safe cache storage
        self._cache = OrderedDict()  # OrderedDict for LRU
        self._current_size_mb = 0.0
        self._hits = 0
        self._misses = 0
    
    def _get_cache_key(self, image_path):
        """Generate cache key from image path"""
        if isinstance(image_path, str):
            return image_path
        return str(image_path)
    
    def _estimate_size_mb(self, image):
        """Estimate image size in MB"""
        if isinstance(image, Image.Image):
            # PIL Image: width * height * channels * bytes_per_pixel
            w, h = image.size
            channels = len(image.getbands()) if hasattr(image, 'getbands') else 3
            size_bytes = w * h * channels * 4  # Assume float32 (4 bytes)
            return size_bytes / (1024 * 1024)
        elif hasattr(image, 'nbytes'):
            # NumPy array
            return image.nbytes / (1024 * 1024)
        else:
            # Fallback: estimate 1MB
            return 1.0
    
    def get(self, image_path):
        """
        Get cached image if available.
        
        Args:
            image_path: Path to image or image identifier
        
        Returns:
            Cached PIL Image or None if not cached
        """
        if not self.enable:
            return None
        
        key = self._get_cache_key(image_path)
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                img = self._cache.pop(key)
                self._cache[key] = img
                self._hits += 1
                return img.copy() if isinstance(img, Image.Image) else img
        
        self._misses += 1
        return None
    
    def put(self, image_path, image):
        """
        Cache an image.
        
        Args:
            image_path: Path to image or image identifier
            image: PIL Image to cache
        """
        if not self.enable:
            return
        
        key = self._get_cache_key(image_path)
        img_size_mb = self._estimate_size_mb(image)
        
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                old_img = self._cache.pop(key)
                old_size = self._estimate_size_mb(old_img)
                self._current_size_mb -= old_size
            
            # Evict LRU items if needed
            while (self._current_size_mb + img_size_mb > self.max_size_mb and 
                   len(self._cache) > 0):
                lru_key, lru_img = self._cache.popitem(last=False)
                lru_size = self._estimate_size_mb(lru_img)
                self._current_size_mb -= lru_size
            
            # Cache the image
            cached_img = image.copy() if isinstance(image, Image.Image) else image.copy()
            self._cache[key] = cached_img
            self._current_size_mb += img_size_mb
    
    def get_or_compute(self, image_path, compute_fn):
        """
        Get cached image or compute and cache it.
        
        Args:
            image_path: Path to image or image identifier
            compute_fn: Function to compute image if not cached
        
        Returns:
            PIL Image (cached or newly computed)
        """
        # Try to get from cache
        cached = self.get(image_path)
        if cached is not None:
            return cached
        
        # Compute it
        img = compute_fn()
        
        # Cache it
        self.put(image_path, img)
        
        return img
    
    def clear(self):
        """Clear all cached images"""
        with self._lock:
            self._cache.clear()
            self._current_size_mb = 0.0
    
    def get_stats(self):
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'cached_images': len(self._cache),
                'size_mb': self._current_size_mb,
                'max_size_mb': self.max_size_mb
            }
    
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print(f"[ImageCache] Hits: {stats['hits']}, Misses: {stats['misses']}, "
              f"Hit Rate: {stats['hit_rate']:.1f}%, "
              f"Cached: {stats['cached_images']} images, "
              f"Size: {stats['size_mb']:.1f} MB / {stats['max_size_mb']} MB")


# Global cache instance (shared across Trainer/Evaluator)
_global_cache = None

def get_global_cache(enable=True, max_size_mb=2048, verbose=False):
    """Get or create global image cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = ImageCache(enable=enable, max_size_mb=max_size_mb, verbose=verbose)
    return _global_cache

def clear_global_cache():
    """Clear global cache"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()

