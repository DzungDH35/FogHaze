import io
import os

import cv2
import numpy as np
from starlette import status

from defog_sr import defog, SuperResolution
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()
sr = SuperResolution()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upscale")
async def upload(
        file: UploadFile = File(...),
        scale: bool = Form(),
        denoise: bool = Form(),
        must_defog: bool = Form()
):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.asarray(bytearray(contents), dtype="uint8"), cv2.IMREAD_COLOR)
        print(scale, denoise, must_defog)
        if scale:
            if image.shape[0] > 1500 or image.shape[1] > 1500:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"mgs": "file too large to scale, pls choose size < 1500x1500"}
                )

            image = sr.upscale_image(image)
        if must_defog:
            image = defog(image, 3)
        if denoise:
            image = cv2.fastNlMeansDenoisingColored(image, None, 3, 10, 7, 21)

        filename, file_extension = os.path.splitext(file.filename)

        _, bts = cv2.imencode(file_extension, image)

        return StreamingResponse(io.BytesIO(bts.tobytes()), media_type=file.content_type)

    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"mgs": e})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
