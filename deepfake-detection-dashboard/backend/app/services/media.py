"""JSON-safe encoding of preview and explanation images."""

import base64

import cv2


def image_to_data_url(image_rgb, jpeg_quality=85, max_width=640):
    height, width = image_rgb.shape[:2]
    if width > max_width:
        scale = max_width / width
        image_rgb = cv2.resize(
            image_rgb,
            (max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    )
    if not ok:
        raise RuntimeError("Could not encode a dashboard preview image.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
