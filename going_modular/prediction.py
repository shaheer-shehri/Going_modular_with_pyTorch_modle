import torch
import torchvision
from timeit import default_timer as timer
import pathlib
from PIL import Image
from tqdm.auto import tqdm
from typing import List,Dict
def make_prediction(paths:List[pathlib.Path],
                    model:torch.nn.Module,
                    transform:torchvision.transforms,
                    class_names:List[str],
                    device:str= "cuda" if torch.cuda.is_available() else "cpu"):
  pred_list = []
  for path in paths:
    pred_dict = {}

    pred_dict["image_path"] = path
    class_name = path.parent.stem
    pred_dict["class_name"] = class_name

    start_time = timer()

    img = Image.open(path)

    transformed_image = transform(img).unsqueeze(dim=0)

    # sending model to target device;
    model.to(device)
    model.eval()

    with torch.inference_mode():
      pred_logit = model(img.to(device))
      pred_prob = torch.softmax(pred_logit, dim=1)
      pred_label = torch.argmax(pred_prob, dim=1)
      pred_class = class_names[pred_label.cup()]
      pred_dict["pred_label"] = class_names[pred_label]
      pred_dict["pred_class"] = pred_class

      end_time = timer()
      pred_dict["time_for_pred"] = round(end_time - start_time, 4)

    pred_dict["correct"] = class_name == pred_class
    pred_list.append(pred_dict)

  return pred_list
