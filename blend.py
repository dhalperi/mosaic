from PIL import Image

if __name__ == "__main__":
    SELECTED = 'output-L-correlation-shiftpct-15.jpg'
    target = Image.open('IMG_3178.JPG').convert('L')
    output = Image.open(f'output/{SELECTED}').convert('L')
    assert target.size == output.size

    for alpha in range(0, 40, 5):
        blended = Image.blend(output, target, alpha / 100)
        blended.save(f'output/blended-{alpha:02d}-{SELECTED}')