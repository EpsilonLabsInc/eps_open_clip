import io
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import boto3
import pyarrow.csv as pa_csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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
        sep="\t",
        tokenizer=None,
        bucket_name="epsilonlabs-datasets-weur",
        prefix="png/512x512/",
        # R2 credentials - set these via environment variables or pass directly
        endpoint_url=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        # Performance tuning
        max_pool_connections=50,
        prefetch_workers=4,
    ):
        logging.debug(f"Loading csv data from {input_filename}.")

        # Load CSV using PyArrow for memory-mapped, zero-copy access
        # This prevents memory duplication across worker processes
        parse_options = pa_csv.ParseOptions(delimiter=sep)
        read_options = pa_csv.ReadOptions(use_threads=True, block_size=2**20)

        self.table = pa_csv.read_csv(
            input_filename, parse_options=parse_options, read_options=read_options
        )

        # Store column references (not copies!)
        # Arrow arrays are memory-mapped and shared across workers
        self.images_col = self.table[img_key]
        self.captions_col = self.table[caption_key]
        self._length = len(self.table)

        self.transforms = transforms
        self.tokenize = tokenizer

        # R2 configuration
        self.bucket_name = bucket_name
        self.prefix = prefix

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

        logging.debug(f"Done loading data. {self._length} samples.")

    def __del__(self):
        """Cleanup resources when dataset is destroyed"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def __len__(self):
        return self._length

    def _fetch_image_from_r2(self, filepath):
        """Fetch image bytes from R2"""
        # Remove leading slash if present
        key = filepath.lstrip("/")
        # Add prefix if not already present
        if not key.startswith(self.prefix):
            key = self.prefix + key

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
        # Zero-copy access to Arrow columns - no refcount modification
        # .as_py() only converts to Python string when needed
        image_path = self.images_col[idx].as_py()
        caption = self.captions_col[idx].as_py()

        image = self._fetch_image_from_r2(str(image_path))
        try:
            images = self.transforms(image)
            texts = self.tokenize([str(caption)])[0]
            return images, texts
        finally:
            # Close PIL image to free memory buffer
            image.close()


def get_r2_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    Drop-in replacement for get_csv_dataset that reads directly from R2.

    Usage in your training script:
        # Replace:
        # data = get_csv_dataset(args, preprocess_train, is_train=True, tokenizer=tokenizer)

        # With:
        data = get_r2_csv_dataset(args, preprocess_train, is_train=True, tokenizer=tokenizer)
    """
    import os

    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    # Get R2 credentials from environment or rclone config
    endpoint_url = os.getenv(
        "R2_ENDPOINT_URL"
    )  # e.g., https://<account_id>.r2.cloudflarestorage.com
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

    # Wrap tokenizer to prevent unbounded cache growth (memory leak fix)
    wrapped_tokenizer = LimitedCacheTokenizer(tokenizer, max_cache_size=10000)

    # Use non-cached version to avoid memory leaks
    dataset = R2CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=wrapped_tokenizer,
        bucket_name=getattr(args, "r2_bucket_name", "epsilonlabs-datasets-weur"),
        prefix=getattr(args, "r2_prefix", "png/512x512/"),
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        max_pool_connections=64,
        prefetch_workers=8,
    )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        prefetch_factor=4,  # Prefetch 4 batches per worker
        persistent_workers=True,  # Safe now with limited tokenizer cache
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    from dataclasses import dataclass

    @dataclass
    class DataInfo:
        dataloader: DataLoader
        sampler: DistributedSampler = None
        shared_epoch: None = None  # Not used for CSV dataset

        def set_epoch(self, epoch):
            """Set epoch for distributed sampler"""
            if self.sampler is not None and isinstance(
                self.sampler, DistributedSampler
            ):
                self.sampler.set_epoch(epoch)

    return DataInfo(dataloader=dataloader, sampler=sampler)
