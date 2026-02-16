from pathlib import Path

import cv2
import numpy as np

from photo_to_scene import generate_scene_from_photo


def main() -> int:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (200, 200), (280, 280), (0, 0, 255), -1)
    cv2.circle(img, (420, 260), 35, (0, 255, 0), -1)

    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    img_path = uploads / "test.jpg"
    cv2.imwrite(str(img_path), img)

    objs, out = generate_scene_from_photo(
        template_xml_path="photo_template.xml",
        image_path=img_path,
        output_xml_path=uploads / "generated_test.xml",
    )
    print("objects:", len(objs))
    print("xml:", out)
    for o in objs:
        print(o.name, o.geom_type, o.pos, o.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

