
VALID_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def parse_path_into_bgr_images(path: str) -> list:
    images = []

    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VALID_IMG_EXTENSIONS:
        img = cv.imread(path)
        if img is None:
            raise Exception('Image cannot be read!')
        images.append(cv.imread(path))
    elif os.path.isdir(path):
        for file_name in os.listdir(path):
            if os.path.splitext(file_name)[1].lower() in VALID_IMG_EXTENSIONS:
                img = cv.imread(os.path.join(path, file_name))
                if img is None:
                    raise Exception('Image cannot be read!')
                images.append(img)
    else:
        raise Exception('Path is not a file or directory!')
    
    return images
