from PIL import Image

# im = Image.open("weights.ppm")
# im.save("weights.jpg")

for i in range(0,1428):
    im = Image.open('weight-{:02d}.ppm'.format(i))
    im.save('weights-{:02d}.jpg'.format(i))