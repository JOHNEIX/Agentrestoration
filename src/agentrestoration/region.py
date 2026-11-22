"""Manual repair region, fixed context crop and lossless outside-region composition."""
import math

from PIL import Image, ImageChops, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .core import write_json

CONTEXT_PIXELS = 32


class RepairRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x0: float = Field(ge=0, lt=1, allow_inf_nan=False)
    y0: float = Field(ge=0, lt=1, allow_inf_nan=False)
    x1: float = Field(gt=0, le=1, allow_inf_nan=False)
    y1: float = Field(gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("修复区域必须有正的宽度和高度")
        return self

    def pixels(self, size):
        w, h = size
        # Normalizing an integer pixel and multiplying back can yield n-1e-14.
        box = (math.floor(self.x0*w+1e-9), math.floor(self.y0*h+1e-9),
               math.ceil(self.x1*w-1e-9), math.ceil(self.y1*h-1e-9))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("修复区域小于一个像素")
        return box


def prepare_region(original, selection, job):
    region = RepairRegion.model_validate(selection)
    with Image.open(original) as image:
        box = region.pixels(image.size)
        x0, y0, x1, y1 = box
        crop_box = (max(0, x0-CONTEXT_PIXELS), max(0, y0-CONTEXT_PIXELS),
                    min(image.width, x1+CONTEXT_PIXELS), min(image.height, y1+CONTEXT_PIXELS))
        image.crop(crop_box).save(job.path / "region_input.png")
        mask = Image.new("L", image.size)
        ImageDraw.Draw(mask).rectangle((x0, y0, x1-1, y1-1), fill=255)
        mask.save(job.path / "region_mask.png")
    data = dict(selection=region.model_dump(), box=list(box), crop_box=list(crop_box),
                context_pixels=CONTEXT_PIXELS, input="region_input.png", mask="region_mask.png",
                mode="manual_rectangle")
    job.data["local_repair"] = data
    write_json(job.path / "region.json", data)
    job.event("region_prepared", **data)
    return data


def region_instruction(data):
    x0, y0, x1, y1 = data["box"]
    cx0, cy0, cx1, cy1 = data["crop_box"]
    target = [(x0-cx0)/(cx1-cx0), (y0-cy0)/(cy1-cy0),
              (x1-cx0)/(cx1-cx0), (y1-cy0)/(cy1-cy0)]
    return ("\nLocal restoration: only restore the selected rectangle with normalized "
            f"[left, top, right, bottom] coordinates {target}. "
            "Surrounding pixels provide context. Preserve the image aspect ratio, composition, "
            "object positions and the selected region's content. Do not zoom, reframe or add objects.")


def compose_region(original, raw, data, job, candidate_id):
    with Image.open(original) as source, Image.open(raw) as restored:
        source, restored = source.convert("RGB"), restored.convert("RGB")
        cx0, cy0, cx1, cy1 = data["crop_box"]
        target_size = (cx1-cx0, cy1-cy0)
        raw_size = restored.size
        # Geometry is checked by the caller before an equal-aspect output is mapped.
        if restored.size != target_size:
            restored = restored.resize(target_size, Image.Resampling.LANCZOS)
        x0, y0, x1, y1 = data["box"]
        result = source.copy()
        result.paste(restored.crop((x0-cx0, y0-cy0, x1-cx0, y1-cy0)), (x0, y0))
        difference = ImageChops.difference(source, result)
        for outside in ((0,0,source.width,y0), (0,y1,source.width,source.height),
                        (0,y0,x0,y1), (x1,y0,source.width,y1)):
            if difference.crop(outside).getbbox() is not None:
                raise ValueError("修复区域外的像素发生变化")
        output = job.path / f"{candidate_id}_composite.png"
        result.save(output)
    metadata = dict(raw_path=raw.name, path=output.name, raw_size=list(raw_size),
                    mapped_size=list(target_size), outside_changed_pixels=0)
    job.event("region_composited", candidate_id=candidate_id, **metadata)
    return output, metadata
