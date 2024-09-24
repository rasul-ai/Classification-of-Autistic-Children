import torch
import cv2
import torchvision.transforms as transforms
import argparse

from resnet import pretrained_model

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', 
    default='/network/scratch/i/islamria/qcontrol/customyolo/asd/data/test/non_autistic/001.jpg',
    help='path to the input image')
args = vars(parser.parse_args())

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# number of classes
labels = ['non_autistic', 'autistic']

model = pretrained_model().to(device)
checkpoint = torch.load(r'/network/scratch/i/islamria/qcontrol/customyolo/asd/outputs/model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# read and preprocess the image
image = cv2.imread(args['input'])
if image is None:
    raise ValueError(f"Unable to read image {args['input']}")

# get the ground truth class from the filename or path
gt_class = args['input'].split('/')[-2]  # Assuming folder names reflect class

# convert to RGB format for preprocessing
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
image = torch.unsqueeze(image, 0)  # add batch dimension

# forward pass to get predictions
with torch.no_grad():
    outputs = model(image.to(device))
output_label = torch.topk(outputs, 1)

# convert the prediction to human-readable class
pred_class = labels[int(output_label.indices)]

# annotate the original image with the predictions
cv2.putText(orig_image, 
    f"GT: {gt_class}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.6, (0, 255, 0), 2, cv2.LINE_AA
)
cv2.putText(orig_image, 
    f"Pred: {pred_class}",
    (10, 55),
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.6, (0, 0, 255), 2, cv2.LINE_AA
)

# save the result
output_path = f"outputs/{gt_class}_{args['input'].split('/')[-1].split('.')[0]}.png"
cv2.imwrite(output_path, orig_image)
print(f"GT: {gt_class}, Pred: {pred_class}. Result saved to {output_path}")
