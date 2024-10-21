import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence

from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

from src.utils.env_utils import DEFAULT_DATA_DIR, GPT_4O_CACHE_DIR
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)
