import csv
import gc
import io
import logging
import math
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import boto3
import pandas as pd
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info


class LRUCache(OrderedDict):
    """
    Efficient LRU cache using OrderedDict with O(1) operations.
    Automatically evicts least recently used items when max size is exceeded.
    """

    def __init__(self, maxsize=10000):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        # Move accessed item to end (most recently used)
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        # Add/update item at end
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        # Evict least recently used if over capacity
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class LimitedCacheTokenizer:
    """
    Wrapper around a tokenizer to limit BPE cache growth and prevent memory leaks.

    The underlying tokenizer's BPE cache grows unbounded during training, causing
    memory leaks in multiprocessing scenarios. This wrapper replaces the cache with
    a proper LRU cache that keeps frequently used tokens while evicting rare ones.
    """

    def __init__(self, tokenizer, max_cache_size=10000):
        self.tokenizer = tokenizer
        self.max_cache_size = max_cache_size
        self._cache_replaced = False

    def __call__(self, *args, **kwargs):
        # Lazy initialization: replace cache on first call in worker process
        if not self._cache_replaced and hasattr(self.tokenizer, "cache"):
            # Replace the unbounded dict with LRU cache
            original_cache = self.tokenizer.cache
            lru_cache = LRUCache(maxsize=self.max_cache_size)
            # Pre-populate with existing entries (e.g., special tokens)
            lru_cache.update(original_cache)
            self.tokenizer.cache = lru_cache
            self._cache_replaced = True

        return self.tokenizer(*args, **kwargs)


class R2CsvDataset(Dataset):
    """Fast CSV dataset that reads images directly from Cloudflare R2.

    Uses PyArrow for zero-copy, memory-mapped CSV access to prevent
    memory duplication across dataloader workers.
    """

    def __init__(
        self,
        input_filename,
        transforms,
        img_key,
        caption_key,
        bucket_name,
        *,
        sep="\t",
        tokenizer=None,
        # R2 credentials - set these via environment variables or pass directly
        endpoint_url=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        # Performance tuning
        max_pool_connections=50,
        prefetch_workers=4,
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.tokenize = tokenizer

        # R2 configuration
        self.bucket_name = bucket_name

        # Initialize S3 client with connection pooling
        from botocore.config import Config

        config = Config(
            max_pool_connections=max_pool_connections,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=config,
        )

        # Thread pool for prefetching
        self.executor = ThreadPoolExecutor(max_workers=prefetch_workers)

        logging.debug(f"Done loading data. {len(self.images)} samples.")

    def __del__(self):
        """Cleanup resources when dataset is destroyed"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def __len__(self):
        return len(self.captions)

    def _fetch_image_from_r2(self, filepath):
        """Fetch image bytes from R2"""
        # Remove leading slash if present
        key = filepath.lstrip("/")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            # Read the body and explicitly close the streaming connection
            body = response["Body"]
            try:
                img_bytes = body.read()
            finally:
                body.close()  # Critical: Close the streaming body to release connection
            return Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            logging.warning(f"Failed to load {key}: {e}")
            # Return a blank image as fallback
            return Image.new("RGB", (512, 512))

    def __getitem__(self, idx):
        image = self._fetch_image_from_r2(str(self.images[idx]))
        try:
            images = self.transforms(image)
            texts = self.tokenize([str(self.captions[idx])])[0]
            return images, texts
        finally:
            # Close PIL image to free memory buffer
            image.close()


class R2CsvIterableDataset(IterableDataset):
    """Memory-efficient CSV dataset that avoids copy-on-write memory duplication.

    Each DataLoader worker independently reads only its assigned portion of the CSV,
    preventing the 18GB * 8 GPUs * 8 workers = 1152GB memory explosion problem.

    Works with distributed training by splitting data across:
    1. DDP ranks (GPU processes)
    2. DataLoader workers within each rank
    """

    def __init__(
        self,
        input_filename,
        transforms,
        img_key,
        caption_key,
        bucket_name,
        *,
        sep="\t",
        tokenizer=None,
        # R2 credentials
        endpoint_url=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        # Performance tuning
        max_pool_connections=50,
        prefetch_workers=4,
        # Chunking for memory efficiency
        chunksize=1000,
    ):
        self.input_filename = input_filename
        self.transforms = transforms
        self.img_key = img_key
        self.caption_key = caption_key
        self.sep = sep
        self.tokenize = tokenizer
        self.chunksize = chunksize

        # R2 configuration
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.max_pool_connections = max_pool_connections
        self.prefetch_workers = prefetch_workers

        # Expand user home directory (~) in path
        self.input_filename = os.path.expanduser(self.input_filename)

        # Count total rows without loading full CSV into memory
        logging.debug(f"Counting rows in {self.input_filename}...")
        with open(self.input_filename) as f:
            self.num_samples = sum(1 for _ in f) - 1  # -1 for header
        logging.debug(f"Found {self.num_samples} samples in CSV.")

    def _init_s3_client(self):
        """Initialize S3 client lazily in worker process (not in main process)"""
        if not hasattr(self, "s3_client"):
            from botocore.config import Config

            config = Config(
                max_pool_connections=self.max_pool_connections,
                retries={"max_attempts": 3, "mode": "adaptive"},
            )

            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                config=config,
            )

            # Thread pool for prefetching (initialized per worker)
            self.executor = ThreadPoolExecutor(max_workers=self.prefetch_workers)

    def _fetch_image_from_r2(self, filepath):
        """Fetch image bytes from R2"""
        self._init_s3_client()

        # Remove leading slash if present
        key = filepath.lstrip("/")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            # Read the body and explicitly close the streaming connection
            body = response["Body"]
            try:
                img_bytes = body.read()
            finally:
                body.close()  # Critical: Close the streaming body to release connection
            return Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            logging.warning(f"Failed to load {key}: {e}")
            # Return None to signal failure, caller will skip this example
            return None

    def __iter__(self):
        """
        Each worker reads only its assigned slice of the CSV file.
        This prevents copy-on-write memory duplication across workers.

        Uses csv.DictReader for true line-by-line streaming without materializing
        large skiprows ranges or holding chunk iterators in memory.
        """
        # Step 1: Determine DDP rank (which GPU process we're in)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Step 2: Determine worker ID within this rank
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # Step 3: Calculate this worker's global ID and slice
        # Example: 8 GPUs × 8 workers = 64 total workers
        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        # Each worker gets approximately num_samples / total_workers rows
        per_worker = int(math.ceil(self.num_samples / total_workers))
        start_idx = global_worker_id * per_worker
        end_idx = min(start_idx + per_worker, self.num_samples)
        num_rows_to_read = end_idx - start_idx

        logging.debug(
            f"Rank {rank}/{world_size}, Worker {worker_id}/{num_workers} "
            f"(global {global_worker_id}/{total_workers}): "
            f"processing rows {start_idx}-{end_idx} ({num_rows_to_read} rows)"
        )

        # Step 4: Read ONLY this worker's portion using csv.DictReader
        # This avoids materializing large skiprows ranges and memory leaks from pandas chunking
        try:
            with open(self.input_filename, "r", newline="") as csvfile:
                # Detect delimiter from first line if using comma
                reader = csv.DictReader(csvfile, delimiter=self.sep)

                # Use islice to efficiently skip to our start position and read only our slice
                # This is memory-efficient: doesn't materialize a huge range object
                rows_to_process = islice(reader, start_idx, end_idx)

                processed_count = 0
                for row in rows_to_process:
                    # Fetch image from R2
                    image = self._fetch_image_from_r2(str(row[self.img_key]))
                    if image is None:
                        # Skip this example if image failed to load
                        continue

                    try:
                        # Transform and tokenize
                        images = self.transforms(image)
                        texts = self.tokenize([str(row[self.caption_key])])[0]
                        yield images, texts
                    finally:
                        # Close PIL image to free memory buffer
                        image.close()

                    processed_count += 1

                    # Periodic garbage collection to release any lingering references
                    if processed_count % self.chunksize == 0:
                        gc.collect()

        except Exception as e:
            logging.error(f"Worker {global_worker_id} encountered error: {e}")
            raise

    def __len__(self):
        """Return total number of samples across all workers"""
        return self.num_samples


def get_r2_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    Memory-efficient CSV dataset that avoids copy-on-write memory duplication.

    Uses IterableDataset where each worker independently reads only its assigned
    portion of the CSV, preventing the 18GB * 8 GPUs * 8 workers = 1152GB
    memory explosion problem.

    Usage in your training script:
        data = get_r2_csv_dataset(args, preprocess_train, is_train=True, tokenizer=tokenizer)
    """
    import os

    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    # Get R2 credentials from environment
    endpoint_url = os.getenv(
        "R2_ENDPOINT_URL"
    )  # e.g., https://<account_id>.r2.cloudflarestorage.com
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

    # Wrap tokenizer to prevent unbounded cache growth (memory leak fix)
    wrapped_tokenizer = LimitedCacheTokenizer(tokenizer, max_cache_size=100)

    # Use IterableDataset to avoid memory duplication across workers
    dataset = R2CsvIterableDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        bucket_name=getattr(args, "r2_streaming_bucket"),
        sep=args.csv_separator,
        tokenizer=wrapped_tokenizer,
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        max_pool_connections=64,
        prefetch_workers=8,
        chunksize=1000,  # Process CSV in 1000-row chunks for memory efficiency
    )

    num_samples = len(dataset)

    # IterableDataset handles distributed splitting internally, no sampler needed
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IterableDataset doesn't support shuffle parameter
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_train,
        prefetch_factor=2,  # Reduced from 4 to 2 to halve prefetch memory usage
        persistent_workers=False,  # Restart workers to clear accumulated memory each epoch
    )

    dataloader.num_samples = num_samples
    # Calculate num_batches manually (can't use len(dataloader) with IterableDataset)
    dataloader.num_batches = num_samples // (args.batch_size * args.world_size)

    from dataclasses import dataclass

    @dataclass
    class DataInfo:
        dataloader: DataLoader
        sampler: None = None  # IterableDataset doesn't use sampler
        shared_epoch: None = None  # Not used for CSV dataset

        def set_epoch(self, epoch):
            """No-op for IterableDataset (no sampler to set epoch on)"""
            pass

    return DataInfo(dataloader=dataloader, sampler=None)
