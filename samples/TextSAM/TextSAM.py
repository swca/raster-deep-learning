# import importlib
import json
import math
import os
import re
import sys
import typing
from enum import Enum
from glob import glob

import arcpy
import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError as e:
    raise RuntimeError(
        "PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials"
    ) from e
try:
    import GPUtil
except ModuleNotFoundError:
    pass


def add_to_sys_path(*paths: str) -> None:
    """Add multiple directories to the system path if not already added."""
    for path in paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Path {full_path} does not exist")
        if full_path not in sys.path:
            sys.path.insert(0, full_path)


add_to_sys_path("segment-anything", "GroundingDINO-main", ".")

from groundingdino.models import build_model  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from groundingdino.util import box_ops  # noqa: E402
from groundingdino.util.inference import predict  # noqa: E402
from groundingdino.datasets import transforms as T  # noqa: E402
from segment_anything import sam_model_registry, SamPredictor  # noqa: E402
from segment_anything.modeling import Sam  # noqa: E402


def get_available_device(max_memory: float = 0.8) -> int:
    """
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    """

    try:
        GPUs = GPUtil.getGPUs()
    except NameError:
        return 0
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available


def get_centroid(polygon: typing.List[typing.List[float]]) -> typing.List[float]:
    """
    Get the centroid of a polygon
    :param polygon: list of points that represent the polygon
    :return: centroid of the polygon
    """
    polygon_array = np.array(polygon)
    return [polygon_array[:, 0].mean(), polygon_array[:, 1].mean()]


def check_centroid_in_center(
    centroid: typing.List[float],
    start_x: int,
    start_y: int,
    chip_sz: int,
    padding: int,
) -> bool:
    """
    Check if the centroid is in the center of the chip
    :param centroid: centroid of the polygon
    :param start_x: start x of the chip
    :param start_y: start y of the chip
    :param chip_sz: size of the chip
    :param padding: padding
    :return: True if the centroid is in the center of the chip, False otherwise
    """
    return (
        (centroid[1] >= (start_y + padding))
        and (centroid[1] <= (start_y + (chip_sz - padding)))
        and (centroid[0] >= (start_x + padding))
        and (centroid[0] <= (start_x + (chip_sz - padding)))
    )


def find_i_j(
    centroid: typing.List[float],
    n_rows: int,
    n_cols: int,
    chip_sz: int,
    padding: int,
    filter_detections: bool = False,
) -> typing.Union[typing.Tuple[int, int, bool], None]:
    """
    Find the i, j index of the centroid in the grid
    :param centroid: centroid of the polygon
    :param n_rows: number of rows in the grid
    :param n_cols: number of cols in the grid
    :param chip_sz: size of the chip
    :param padding: padding
    :param filter_detections: whether to filter the detections
    :return: i, j index of the centroid in the grid, and whether the centroid is in the center of the chip
    """
    for i in range(n_rows):
        for j in range(n_cols):
            start_x = i * chip_sz
            start_y = j * chip_sz

            if (
                (centroid[1] > start_y)
                and (centroid[1] < (start_y + chip_sz))
                and (centroid[0] > start_x)
                and (centroid[0] < (start_x + chip_sz))
            ):
                in_center = check_centroid_in_center(
                    centroid, start_x, start_y, chip_sz, padding
                )
                if filter_detections:
                    if in_center:
                        return i, j, in_center
                else:
                    return i, j, in_center
    return None


def calculate_rectangle_size_from_batch_size(batch_size: int) -> typing.Tuple[int, int]:
    """
    calculate number of rows and cols to composite a rectangle given a batch size
    :param batch_size:
    :return: number of cols and number of rows
    """
    rectangle_height = int(math.sqrt(batch_size) + 0.5)
    rectangle_width = int(batch_size / rectangle_height)

    if rectangle_height * rectangle_width > batch_size:
        if rectangle_height >= rectangle_width:
            rectangle_height = rectangle_height - 1
        else:
            rectangle_width = rectangle_width - 1

    if (rectangle_height + 1) * rectangle_width <= batch_size:
        rectangle_height = rectangle_height + 1
    if (rectangle_width + 1) * rectangle_height <= batch_size:
        rectangle_width = rectangle_width + 1

    # swap col and row to make a horizontal rect
    if rectangle_height > rectangle_width:
        rectangle_height, rectangle_width = rectangle_width, rectangle_height

    if rectangle_height * rectangle_width != batch_size:
        return batch_size, 1

    return rectangle_height, rectangle_width


def get_tile_size(
    model_height: int,
    model_width: int,
    padding: int,
    batch_height: int,
    batch_width: int,
) -> typing.Tuple[int, int]:
    """
    Calculate request tile size given model and batch dimensions
    :param model_height:
    :param model_width:
    :param padding:
    :param batch_width:
    :param batch_height:
    :return: tile height and tile width
    """
    tile_height = (model_height - 2 * padding) * batch_height
    tile_width = (model_width - 2 * padding) * batch_width

    return tile_height, tile_width


def tile_to_batch(
    pixel_block: np.ndarray[typing.Any, typing.Any],
    model_height: int,
    model_width: int,
    padding: int,
    fixed_tile_size: bool = True,
    **kwargs: typing.Any,
) -> typing.Tuple[np.ndarray[typing.Any, typing.Any], int, int]:
    """
    Convert pixel block to batch
    :param pixel_block: pixel block
    :param model_height: model height
    :param model_width: model width
    :param padding: padding
    :param fixed_tile_size: whether to use fixed tile size
    :param kwargs: other parameters
    :return: batch, batch height, batch width
    """
    inner_width = model_width - 2 * padding
    inner_height = model_height - 2 * padding

    band_count, pb_height, pb_width = pixel_block.shape
    pixel_type = pixel_block.dtype

    if fixed_tile_size is True:
        batch_height = kwargs["batch_height"]
        batch_width = kwargs["batch_width"]
    else:
        batch_height = math.ceil((pb_height - 2 * padding) / inner_height)
        batch_width = math.ceil((pb_width - 2 * padding) / inner_width)

    batch = np.zeros(
        shape=(batch_width * batch_height, band_count, model_height, model_width),
        dtype=pixel_type,
    )
    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        # pixel block might not be the shape (band_count, model_height, model_width)
        sub_pixel_block = pixel_block[
            :,
            y * inner_height : y * inner_height + model_height,
            x * inner_width : x * inner_width + model_width,
        ]
        sub_pixel_block_shape = sub_pixel_block.shape
        batch[
            b, :, : sub_pixel_block_shape[1], : sub_pixel_block_shape[2]
        ] = sub_pixel_block

    return batch, batch_height, batch_width


class TextSAM:
    """TextSAM class"""

    fields: typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]] = {
        "fields": [
            {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
            {"name": "Class", "type": "esriFieldTypeString", "alias": "Class"},
            {
                "name": "Confidence",
                "type": "esriFieldTypeDouble",
                "alias": "Confidence",
            },
            {"name": "Shape", "type": "esriFieldTypeGeometry", "alias": "Shape"},
        ]
    }
    """Fields for the output feature class"""

    class GeometryType(Enum):
        """Geometry type enumeration for the output feature class."""

        Point: int = 1
        Multipoint: int = 2
        Polyline: int = 3
        Polygon: int = 4

    def __init__(self) -> None:
        """Constructor for TextSAM class."""

        self.name: str = "Text SAM Model"
        """Name of the python raster function"""

        self.description: str = "This python raster function applies computer vision to segment anything from text input"
        """Description of the python raster function"""

        self.features: dict[
            str,
            typing.Union[
                str,
                typing.Dict[str, str],
                typing.List[typing.Dict[str, typing.Any]],
                typing.List[typing.Dict[str, typing.Any]],
            ],
        ] = {
            "displayFieldName": "",
            "fieldAliases": {
                "FID": "FID",
                "Class": "Class",
                "Confidence": "Confidence",
            },
            "geometryType": "esriGeometryPolygon",
            "fields": [
                {"name": "FID", "type": "esriFieldTypeOID", "alias": "FID"},
                {"name": "Class", "type": "esriFieldTypeString", "alias": "Class"},
                {
                    "name": "Confidence",
                    "type": "esriFieldTypeDouble",
                    "alias": "Confidence",
                },
            ],
            "features": [],
        }
        """Features for the output feature class"""

        self.json_info: typing.Dict[str, typing.Any]
        """JSON info for the model"""

        self.mask_generator: typing.Any
        """Mask generator for the model"""

        self.device_id: typing.Union[int, str, None]
        """Device ID for the model"""

        self.groundingdino_model: typing.Any
        """GroundingDINO model"""

        self.tytx: int
        """Tile size"""

        self.batch_size: int
        """Batch size"""

        self.padding: int
        """Padding"""

        self.text_prompt: str
        """Text prompt"""

        self.box_threshold: float
        """Box threshold"""

        self.text_threshold: float
        """Text threshold"""

    @staticmethod
    def get_sam(sam_root_dir: typing.Optional[str] = None) -> Sam:
        """
        Get the SAM model
        :param sam_root_dir: SAM root directory
        :return: SAM model
        """
        _sam_root_dir = sam_root_dir or os.path.join(
            os.path.dirname(__file__), "segment-anything"
        )
        if not os.path.exists(_sam_root_dir):
            raise FileNotFoundError("SAM root directory not found")
        # loading the SAM model checkpoint and initliazing SAM mask_generator
        sam_checkpoints = glob(os.path.join(_sam_root_dir, "models", "sam_vit_*.pth"))
        if len(sam_checkpoints) == 0:
            raise FileNotFoundError("SAM model checkpoint not found")
        elif len(sam_checkpoints) > 1:
            raise RuntimeError("Multiple SAM model checkpoints found")
        sam_checkpoint = os.path.abspath(sam_checkpoints[0])
        sam_checkpoint_basename = os.path.basename(sam_checkpoint)
        model_type_pattern = re.compile(r"sam_(vit_[blh])_[0-9a-f]{6}.pth")
        model_type_match = model_type_pattern.match(sam_checkpoint_basename)
        if model_type_match is None:
            raise RuntimeError("Invalid SAM model checkpoint")
        model_type = model_type_match.group(1)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        return sam

    @staticmethod
    def get_cache_and_config(
        gdino_root_dir: typing.Optional[str] = None,
    ) -> typing.Tuple[str, str]:
        """
        Get the cache and config files for GroundingDINO
        :param gdino_root_dir: GroundingDINO root directory
        :return: cache and config files
        """
        _gdino_root_dir = gdino_root_dir or os.path.join(
            os.path.dirname(__file__), "GroundingDINO-main"
        )
        if not os.path.exists(_gdino_root_dir):
            raise FileNotFoundError("GroundingDINO root directory not found")
        # loading the GroundingDINO model checkpoint and config file and initliazing GroundingDINO
        model_dir = os.path.join(_gdino_root_dir, "models")
        if not os.path.exists(model_dir):
            raise FileNotFoundError("GroundingDINO models directory not found")
        cache_file = os.path.join(model_dir, "groundingdino_swinb_cogcoor.pth")
        if not os.path.exists(cache_file):
            raise FileNotFoundError("GroundingDINO model checkpoint not found")
        cache_config_file = os.path.join(model_dir, "GroundingDINO_SwinB.cfg.py")
        if not os.path.exists(cache_config_file):
            raise FileNotFoundError("GroundingDINO model config file not found")
        return cache_file, cache_config_file

    def initialize(self, **kwargs: typing.Any) -> None:
        """
        Initialize the model
        :param kwargs: keyword arguments
        :return: None
        """
        if "model" not in kwargs:
            return

        model = kwargs["model"]
        # model_as_file = True
        try:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                # model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        self.device_id = None
        if "device" in kwargs:
            device = kwargs["device"]
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
                self.device_id = device
            else:
                arcpy.env.processorType = "CPU"
                self.device_id = "cpu"

        sam = self.get_sam()
        sam.to(device=self.device_id)
        self.mask_generator = SamPredictor(sam)

        cache_file, cache_config_file = self.get_cache_and_config()
        args = SLConfig.fromfile(cache_config_file)
        self.groundingdino_model = build_model(args)
        self.groundingdino_model.to(device=self.device_id)
        checkpoint = torch.load(cache_file, map_location="cpu")
        self.groundingdino_model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        self.groundingdino_model.eval()

    def getParameterInfo(self) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Get parameter info
        :return: list of parameter info
        """
        required_parameters = [
            {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Raster",
                "description": "Input Raster",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "device",
                "dataType": "numeric",
                "required": False,
                "displayName": "Device ID",
                "description": "Device ID",
            },
        ]
        required_parameters.extend(
            [
                {
                    "name": "text_prompt",
                    "dataType": "string",
                    "required": False,
                    "value": "",
                    "displayName": "Text Prompt",
                    "description": "Text Prompt",
                },
                {
                    "name": "padding",
                    "dataType": "numeric",
                    "value": int(self.json_info["ImageHeight"]) // 4,
                    "required": False,
                    "displayName": "Padding",
                    "description": "Padding",
                },
                {
                    "name": "batch_size",
                    "dataType": "numeric",
                    "required": False,
                    "value": 4,
                    "displayName": "Batch Size",
                    "description": "Batch Size",
                },
                {
                    "name": "box_threshold",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.2,
                    "displayName": "Box Threshold",
                    "description": "Box Threshold",
                },
                {
                    "name": "text_threshold",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.2,
                    "displayName": "Text Threshold",
                    "description": "Text Threshold",
                },
                {
                    "name": "box_nms_thresh",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.7,
                    "displayName": "box_nms_thresh",
                    "description": "The box IoU cutoff used by non-maximal suppression to filter duplicate masks.",
                },
            ]
        )
        return required_parameters

    def getConfiguration(self, **scalars: typing.Any) -> dict[str, typing.Any]:
        """
        Get configuration
        :param scalars: scalar parameters
        :return: configuration
        """
        self.tytx = int(scalars.get("tile_size", self.json_info["ImageHeight"]))
        self.batch_size = int(math.sqrt(int(scalars.get("batch_size", 4)))) ** 2
        self.padding = int(scalars.get("padding", self.tytx // 4))
        self.text_prompt: str = scalars.get("text_prompt")  # type: ignore

        self.box_threshold = float(scalars.get("box_threshold"))  # type: ignore
        self.text_threshold = float(scalars.get("text_threshold"))  # type: ignore
        (
            self.rectangle_height,
            self.rectangle_width,
        ) = calculate_rectangle_size_from_batch_size(self.batch_size)
        ty, tx = get_tile_size(
            self.tytx,
            self.tytx,
            self.padding,
            self.rectangle_height,
            self.rectangle_width,
        )

        return {
            "inputMask": True,
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "batch_size": self.batch_size,
            "tx": tx,
            "ty": ty,
            "fixedTileSize": 1,
        }

    @classmethod
    def getFields(cls) -> str:
        """
        Get fields
        :return: fields
        """
        return json.dumps(cls.fields)

    @classmethod
    def getGeometryType(cls) -> int:
        """
        Get geometry type
        :return: geometry type
        """
        return cls.GeometryType.Polygon.value

    def vectorize(self, **pixelBlocks: typing.Any) -> dict[str, typing.Any]:
        """
        Vectorize the pixel blocks
        :param pixelBlocks: pixel blocks
        :return: vectorized pixel blocks
        """
        raster_mask = pixelBlocks["raster_mask"]
        raster_pixels = pixelBlocks["raster_pixels"]
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks["raster_pixels"] = raster_pixels

        # create batch from pixel blocks
        batch, batch_height, batch_width = tile_to_batch(
            raster_pixels,
            self.tytx,
            self.tytx,
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )

        mask_list = []
        score_list = []

        # iterate over batch and get segment from model
        for batch_idx, input_pixels in enumerate(batch):
            side = int(math.sqrt(self.batch_size))
            i, j = batch_idx // side, batch_idx % side
            input_pixels = np.moveaxis(input_pixels, 0, -1)
            # for input_pixels in batch:
            pil_image = Image.fromarray(input_pixels)
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_transformed, _ = transform(pil_image, None)
            if "," in self.text_prompt:
                split_prompts = self.text_prompt.split(",")
                cleaned_items = [split_prompt.strip() for split_prompt in split_prompts]
                final_caption = " . ".join(cleaned_items)
            else:
                final_caption = self.text_prompt
            try:
                boxes, logits, phrases = predict(
                    model=self.groundingdino_model,
                    image=image_transformed,
                    caption=final_caption,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device_id,
                )
            except RuntimeError as e:
                if "no elements" in str(e):
                    continue
            W, H = pil_image.size
            boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            updated_boxes = []
            updated_scores = []
            for en1, box1 in enumerate(boxes):
                box1 = box1.numpy()
                x, y, x1, y1 = box1
                w = x1 - x
                h = y1 - y
                if w < 0.25 * int(self.tytx) and h < int(0.25 * self.tytx):
                    updated_boxes.append(boxes[en1])
                    updated_scores.append(logits[en1])
            self.mask_generator.set_image(input_pixels)

            if updated_boxes:
                transformed_boxes = self.mask_generator.transform.apply_boxes_torch(
                    torch.stack(updated_boxes), input_pixels.shape[:2]
                )
                masks, _, _ = self.mask_generator.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(self.device_id),
                    multimask_output=False,
                )
                for counter, mask_value in enumerate(masks):
                    masked_image = mask_value * 1
                    masked_image = masked_image.cpu().numpy()
                    contours, hierarchy = cv2.findContours(
                        (masked_image[0]).astype(np.uint8),
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE,
                        offset=(0, 0),
                    )
                    hierarchy = hierarchy[0]
                    for c_idx, contour in enumerate(contours):
                        contour = contours[c_idx] = contour.squeeze(1)
                        contours[c_idx][:, 0] = contour[:, 0] + (j * self.tytx)
                        contours[c_idx][:, 1] = contour[:, 1] + (i * self.tytx)
                    for (
                        contour_idx,
                        (next_contour, prev_contour, child_contour, parent_contour),
                    ) in enumerate(hierarchy):
                        if parent_contour == -1:
                            coord_list = [contours[contour_idx].tolist()]
                            while child_contour != -1:
                                coord_list.append(contours[child_contour].tolist())
                                child_contour = hierarchy[child_contour][0]
                            mask_list.append(coord_list)
                            score_list.append(
                                str(updated_scores[counter].numpy() * 100)
                            )

        n_rows = int(math.sqrt(self.batch_size))
        n_cols = int(math.sqrt(self.batch_size))
        padding = self.padding
        keep_masks = []
        keep_scores = []

        for idx, mask in enumerate(mask_list):
            if not mask:
                continue
            centroid = get_centroid(mask[0])
            tytx = self.tytx
            grid_location = find_i_j(centroid, n_rows, n_cols, tytx, padding, True)
            if grid_location is not None:
                i, j, in_center = grid_location
                for poly_id, polygon in enumerate(mask):
                    polygon = np.array(polygon)
                    polygon[:, 0] = (
                        polygon[:, 0] - (2 * i + 1) * padding
                    )  # Inplace operation
                    polygon[:, 1] = (
                        polygon[:, 1] - (2 * j + 1) * padding
                    )  # Inplace operation
                    mask[poly_id] = polygon.tolist()
                if in_center:
                    keep_masks.append(mask)
                    keep_scores.append(score_list[idx])

        final_masks = keep_masks
        pred_score = keep_scores

        for mask_idx, final_mask in enumerate(final_masks):
            self.features["features"].append(  # type: ignore
                {
                    "attributes": {
                        "OID": mask_idx + 1,
                        "Class": "Segment",
                        "Confidence": pred_score[mask_idx],
                    },
                    "geometry": {"rings": final_mask},
                }
            )
        return {"output_vectors": json.dumps(self.features)}
