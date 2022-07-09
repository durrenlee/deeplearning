# normalize an image
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

transforms_composer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
]
)

# load the image
img_path = 'Koalainputimage.jpg'
img = Image.open(img_path)

# get normalized image
normalized_img = transforms_composer(img)
# print(normalized_img)
# print(normalized_img.shape)

# convert normalized image to numpy array
np_img = np.array(normalized_img)
# print(np_img)
# print(np_img.shape)

# plot the pixel values
plt.hist(np_img.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
plt.show()

print(np_img.shape)
# transpose from shape of (3,,) to shape of (,,3)
transpose_img = np_img.transpose(1, 2, 0)
# print(transpose_img.shape)
# print(transpose_img)
plt.imshow(transpose_img)
plt.show()