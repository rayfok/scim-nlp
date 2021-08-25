import os
import json
from typing import List, NamedTuple, Union, Dict, Any


class COCOBuilder:
    class CategoryTemplate(NamedTuple):
        id: int
        name: str
        supercategory: str = None

    class ImageTemplate(NamedTuple):
        id: int
        file_name: str
        height: Union[float, int]
        width: Union[float, int]
        paper_id: int
        page_number: int

    class AnnoTemplate(NamedTuple):
        id: int
        bbox: List
        image_id: int
        category_id: int
        area: Union[float, int]
        iscrowd: bool = False

    def __init__(
        self, save_path: str, annotation_name: str, categories: Dict[int, str]
    ):

        self.annotation_name = annotation_name

        self.save_path = save_path
        self.save_path_image = os.path.join(save_path, "images")
        if not os.path.exists(self.save_path_image):
            os.makedirs(self.save_path_image)

        self._categories = self.create_coco_categories(categories)
        self._images = []
        self._annotations = []
        self.cat2id = {cat: int(idx) for idx, cat in categories.items()}

    def add_annotation(
        self, page_name: str, page_image: "Image", page_annotations: "pd.DataFrame"
    ):

        # Firstly Save the image
        image_filename = page_name + ".jpg"
        if not os.path.exists(f"{self.save_path_image}/{image_filename}"):
            page_image.save(f"{self.save_path_image}/{image_filename}")

        # Save the image-level information
        width, height = page_image.size
        *paper_id, page_number = page_name.split("-")
        image_id = len(self._images)

        self._images.append(
            self.ImageTemplate(
                id=image_id,
                file_name=image_filename,
                height=height,
                width=width,
                paper_id="-".join(paper_id),
                page_number=page_number,
            )._asdict()
        )

        # Save the actual annotations
        start_anno_id = len(self._annotations)
        page_annotations = page_annotations.reset_index(drop=True)
        page_annotations["w"] = page_annotations["x_2"] - page_annotations["x_1"]
        page_annotations["h"] = page_annotations["y_2"] - page_annotations["y_1"]
        page_annotations["area"] = page_annotations["w"] * page_annotations["h"]
        page_annotations["category"] = page_annotations["category"].map(self.cat2id)
        annotations = page_annotations.apply(
            lambda row: self.AnnoTemplate(
                id=row.name + start_anno_id,
                bbox=[row["x_1"], row["y_1"], row["w"], row["h"]],
                category_id=row["category"],
                image_id=image_id,
                area=row["area"],
            )._asdict(),
            axis=1,
        ).tolist()
        self._annotations.extend(annotations)

    def create_coco_categories(self, categories: Dict) -> List[Dict]:
        return [
            self.CategoryTemplate(int(idx), category)._asdict()
            for idx, category in categories.items()
        ]

    def export(self):

        coco_json = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": self._categories,
        }

        with open(f"{self.save_path}/{self.annotation_name}.json", "w") as fp:
            json.dump(coco_json, fp)