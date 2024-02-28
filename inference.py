"""
inference.py
"""
from PIL import Image
from torchfusion_utils.models import load_model, save_model
import torch
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import cv2
from PIL import Image
import glob

'''
since the pathways are dependent on the user & the setup they are running,
i am making a comment here to store the different systems
'''
checkpoints = [r'C:\Users\alexx\OneDrive\Senior Design\fire-flame.pt',
               r"C:\Users\aaron\PycharmProjects\Fireflyghter\fire-flame.pt"]
images = [r"C:\Users\alexx\OneDrive\Senior Design\content\img_folder\*",
          r"C:\Users\alexx\PycharmProjects\Fireflyghter\img_folder\*"]
outputs = [r"C:\Users\alexx\OneDrive\Senior Design\outputimg{}",
           r"C:\Users\aaron\PycharmProjects\Fireflyghter\outputimg{}"]

model = models.mobilenet_v3_small(weights=False)
checkpoint_path = checkpoints[1]
load_model(model, checkpoint_path)
model.eval()
load_saved_model = torch.load('fire-flame.pt')
transformer = transforms.Compose([transforms.Resize(225),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5],
                                                       [0.5, 0.5, 0.5])])

for img_path in glob.glob(images[1]):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img.astype('uint8'))
    # orig = img.copy()
    img_processed = transformer(img).unsqueeze(0)
    img_var = Variable(img_processed, requires_grad=False)
    if torch.cuda.is_available():
        img_var = img_var.cuda()
        model.cuda()

    load_saved_model.eval()
    # logp = load_model(img_var)
    expp = torch.softmax(model, dim=1)
    confidence, clas = expp.topk(1, dim=1)

    co = confidence.item() * 100

    class_no = str(clas).split(',')[0]
    class_no = class_no.split('(')
    class_no = class_no[1].rstrip(']]')
    class_no = class_no.lstrip('[[')

    orig = np.array(orig)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (800, 500))

    if class_no == '1':
        label = "Neutral: " + str(co) + "%"
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif class_no == '2':
        label = "Smoke: " + str(co) + "%"
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    elif class_no == '0':
        label = "Fire: " + str(co) + "%"
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(outputs[0].format(img_path.split("/")[-1]), orig),

# import torch
# from torchvision import transforms, models
# from PIL import Image
# import cv2
# import glob
# from torchfusion_utils.fp16 import convertToFP16
# from torchfusion_utils.initializers import *
# from torchfusion_utils.metrics import Accuracy
# from torchfusion_utils.models import load_model,save_model
# import numpy as np


# model_path = r'C:\Users\alexx\OneDrive\Senior Design\fire-flame.pt'

# model = models.resnet50(pretrained=False)
# load_saved_model = torch.load('fire-flame.pt')

# model = load_model(model, model_path)
# model.eval() 

# # Image transformations
# transformer = transforms.Compose([
#     transforms.Resize(225),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# for img_path in glob.glob(r'C:\Users\alexx\OneDrive\Senior Design\content\img_folder\*'):
#     img = Image.open(img_path).convert('RGB')
#     img_transformed = transformer(img).unsqueeze(0)

#     # Perform classification
#     with torch.no_grad():
#         outputs = model(img_transformed)
#         _, predicted = torch.max(outputs, 1)
#         class_no = str(predicted.item())

#     # Convert PIL image to CV2 format for labeling and saving
#     orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#     # Apply classification label based on class_no
#     if class_no == '1':
#         label = "Neutral: {}%".format(outputs.max(1)[0].item() * 100)
#         cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#     elif class_no == '2':
#         label = "Smoke: {}%".format(outputs.max(1)[0].item() * 100)
#         cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     elif class_no == '0':
#         label = "Fire: {}%".format(outputs.max(1)[0].item() * 100)
#         cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     output_path = r"C:\Users\alexx\OneDrive\Senior Design\outputimg\{}".format(img_path.split("\\")[-1])
#     cv2.imwrite(output_path, orig)

# print("Processing complete.")
