from PIL import Image

for i in range(10):
    im = Image.open("circle-0" + str(i) + ".ppm")
    im.save("circle-0" + str(i) + ".jpg")