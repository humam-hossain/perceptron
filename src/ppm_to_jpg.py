from PIL import Image

# im = Image.open("weights.ppm")
# im.save("weights.jpg")

for i in range(1,38):
    im = Image.open('weights-{:02d}.ppm'.format(i))
    im.save('weights-{:02d}.jpg'.format(i))