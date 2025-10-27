import io
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import boto3
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class R2CsvDataset(Dataset):
    """
    Fast CSV dataset that reads images directly from Cloudflare R2.
    Much faster than rclone mount due to:
    - Direct S3 API calls (no FUSE overhead)
    - Parallel prefetching
    - Connection pooling
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
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
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

        logging.debug(f"Done loading data. {len(self.images)} samples.")

    def __len__(self):
        return len(self.captions)

    def _fetch_image_from_r2(self, filepath):
        """Fetch image bytes from R2"""
        # Remove leading slash if present
        key = filepath.lstrip("/")
        # Add prefix if not already present
        if not key.startswith(self.prefix):
            key = self.prefix + key

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            img_bytes = response["Body"].read()
            return Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            logging.warning(f"Failed to load {key}: {e}")
            # Return a blank image as fallback
            return Image.new("RGB", (512, 512))

    def __getitem__(self, idx):
        image = self._fetch_image_from_r2(str(self.images[idx]))
        images = self.transforms(image)
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class CachedR2CsvDataset(R2CsvDataset):
    """
    R2 dataset with LRU cache for frequently accessed images.
    Good for multiple epochs or when some images are accessed multiple times.
    """

    def __init__(self, *args, cache_size=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        # Create a cached version of the fetch function
        self._cached_fetch = lru_cache(maxsize=cache_size)(self._fetch_image_from_r2)

    def __getitem__(self, idx):
        image = self._cached_fetch(str(self.images[idx]))
        images = self.transforms(image)
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


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

    # Use cached version for better performance across epochs
    dataset = CachedR2CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        bucket_name=getattr(args, "r2_bucket_name", "epsilonlabs-datasets-weur"),
        prefix=getattr(args, "r2_prefix", "png/512x512/"),
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        max_pool_connections=100,  # High connection pool for 12 workers
        prefetch_workers=8,
        cache_size=20000,  # Cache 20k images (adjust based on RAM)
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
        persistent_workers=True,  # Keep workers alive between epochs
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    from dataclasses import dataclass

    @dataclass
    class DataInfo:
        dataloader: DataLoader
        sampler: DistributedSampler = None

    return DataInfo(dataloader, sampler)
