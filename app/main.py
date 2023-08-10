from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from foghaze_removal.msialdcp import defoghaze, DEFAULT_PATCH_SIZE, DEFAULT_OMEGA, DEFAULT_T0, DEFAULT_RADIUS, DEFAULT_EPS, DEFAULT_FUSION_WEIGHT, DEFAULT_AL_RESIZE_FACTOR
from starlette import status
import cv2 as cv
import io
import numpy as np
import os
import utilities.utilities as utils
import uvicorn


app = FastAPI()


def white_balance(image):
    image_float = image.astype(np.float32) / 255.0

    # Calculate the average color of the image
    avg_color = np.mean(image_float, axis=(0, 1))

    # Compute the scaling factors for each color channel
    gray_world = np.mean(avg_color)
    scaling_factors = gray_world / avg_color

    # Apply the scaling factors to each color channel
    balanced_image_float = image_float * scaling_factors
    
    balanced_image_float = np.clip(balanced_image_float, 0, 1)

    return (balanced_image_float * 255).astype(np.uint8)


@app.post("/defoghaze")
async def upload(
    file: UploadFile = File(...),
    patch_size: int = Form(default=DEFAULT_PATCH_SIZE),
    omega: float = Form(default=DEFAULT_OMEGA),
    t0: float = Form(default=DEFAULT_T0),
    radius: float = Form(default=DEFAULT_RADIUS),
    epsilon: float = Form(default=DEFAULT_EPS),
    fusion_weight: float = Form(default=DEFAULT_FUSION_WEIGHT),
    al_resize_factor: float = Form(default=DEFAULT_AL_RESIZE_FACTOR),
    post_processing: bool = Form(default=True),
):
    print(patch_size)
    print(al_resize_factor)
    print(post_processing)
    try:
        contents = await file.read()
        image = cv.imdecode(np.asarray(bytearray(contents), dtype="uint8"), cv.IMREAD_COLOR)
        dfh_results = defoghaze(image, patch_size, omega, t0, radius, epsilon, fusion_weight, al_resize_factor)
        image = utils.minmax_normalize(dfh_results['recovered_bgr'], (0, 1), (0, 255), np.uint8)

        if post_processing:
            image = white_balance(image)
            image = cv.fastNlMeansDenoisingColored(image, hColor=20, h=5)

        filename, file_extension = os.path.splitext(file.filename)

        _, bts = cv.imencode(file_extension, image)

        return StreamingResponse(io.BytesIO(bts.tobytes()), media_type=file.content_type)

    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"mgs": e})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
