from PIL import Image

if __name__ == "__main__":
    target = Image.open('IMG_3178.JPG').convert('L')
    output = Image.open('output/output-L.jpg').convert('L')
    assert target.size == output.size

    for alpha in range(0, 100, 10):
        blended = Image.blend(output, target, alpha / 100)
        blended.save(f'blended-{alpha}.jpg')