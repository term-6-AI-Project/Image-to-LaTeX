from ultralytics import YOLO
import torchvision
import os
import torchvision
from torch.utils.data import Dataset
import json
import os
from .char_encoding import visible_char_encoding, full_char_encoding

from tqdm import tqdm
def convert_image_to_binary(image, thresh):
    """Convert image to black and white, which will be referred to as a binary image"""
    fn = lambda x : 1 if x <= thresh else 0
    binary_image = image.convert('L').point(fn, mode='1')
    return binary_image

def create_bounding_box_labels(input_json_file):
    """
    for each image, create a file listing the coordinates of bounding boxes of latex chars of the image
    """
    data = []
    with open(f"{input_json_file}", 'r') as f:
        data = json.load(f)
    data = list(data)
    data.sort(key = lambda x: x["uuid"])
    bounding_box_dict = {}
    for d in data:
        # get output file name
        file_name = f"{d['uuid']}.jpg"
        # extract coordinates from each item in json array
        xmins = d["image_data"]["xmins"]
        ymins = d["image_data"]["ymins"]
        xmaxs = d["image_data"]["xmaxs"]
        ymaxs = d["image_data"]["ymaxs"]
        # make list of bounding box coordinates for each LaTeX character
        bounding_box_dict[file_name] = [[xmin,ymin,xmax,ymax]
                             for xmin,ymin,xmax,ymax in zip(xmins, ymins, xmaxs, ymaxs)]
    return bounding_box_dict

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def get_visible_chars_in_image(image_data):
    """
    Returns
    ---
    char_set set[str]: set of latex chars for the image corresponding to image_data
    """
    char_set = set(image_data.get("visible_latex_chars"))
    return char_set

def get_full_chars_in_image(image_data):
    """
    Returns
    ---
    char_set set[str]: set of latex chars for the image corresponding to image_data
    """
    char_set = set(image_data.get("full_latex_chars"))
    return char_set

def create_visible_char_labels(input_json_file, subset_start = None, subset_end = None):
    """
    input_json_file list[str]: file path of json ground truth for a batch
    file_subset list[str] | None: list of files that need to be checked
    Returns
    ---
    char_set set[str]: set of latex chars that occur in the files that were checked
    char_dict dict[str,dict[str]]: for each file in the files checked, a dict of visible latex chars for that file is returned
    """
    data = []
    with open(f"{input_json_file}", 'r') as f:
        data = json.load(f)
    data = list(data)
    data.sort(key = lambda x: x["uuid"])
    char_set = set()
    char_dict = {}
    if subset_start is not None and subset_end is not None:
        data = [
            d
            for i,d in enumerate(data)
            if i >= subset_start
            and i < subset_end
        ]
    print('create_char_labels',len(data))
    for i,d in enumerate(tqdm(data)):
        # get output file name
        file_name = f"{d['uuid']}.jpg"
        chars_in_image = get_visible_chars_in_image(d["image_data"])
        char_set = char_set.union(chars_in_image)
        char_dict[file_name] = d["image_data"]["visible_latex_chars"]
    return char_set, char_dict

def create_full_char_labels(input_json_file, subset_start = None, subset_end = None):
    """
    input_json_file list[str]: file path of json ground truth for a batch
    file_subset list[str] | None: list of files that need to be checked
    Returns
    ---
    char_set set[str]: set of latex chars that occur in the files that were checked
    char_dict dict[str,dict[str]]: for each file in the files checked, a dict of visible latex chars for that file is returned
    """
    data = []
    with open(f"{input_json_file}", 'r') as f:
        data = json.load(f)
    data = list(data)
    data.sort(key = lambda x: x["uuid"])
    char_set = set()
    char_dict = {}
    if subset_start is not None and subset_end is not None:
        data = [
            d
            for i,d in enumerate(data)
            if i >= subset_start
            and i < subset_end
        ]
    print('create_char_labels',len(data))
    for i,d in enumerate(tqdm(data)):
        # get output file name
        file_name = f"{d['uuid']}.jpg"
        chars_in_image = get_full_chars_in_image(d["image_data"])
        char_set = char_set.union(chars_in_image)
        char_dict[file_name] = d["image_data"]["full_latex_chars"]
    return char_set, char_dict

def get_complete_visible_char_set():
  batch_char_set = []
  batch_char_set.append([])
  complete_char_set = set()
  for i in range(1,11):
    input_json_file_name = f"raw_train_data/batch_{i}/JSON/kaggle_data_{i}.json"
    batch_dir = ""
    cur_batch_char_set, cur_batch_char_dict = create_visible_char_labels(input_json_file_name,subset_start = 0, subset_end = 5000)
    cur_batch_char_list = list(cur_batch_char_set)
    cur_batch_char_list.sort()
    batch_char_set.append(cur_batch_char_set)
    print(f"batch {i}")
    print(f"number of unique LaTeX chars in batch {i}:{len(cur_batch_char_list)}")
    print(f"unique LaTeX chars",cur_batch_char_list)

  for c in batch_char_set:
    complete_char_set = complete_char_set.union(c)
  complete_char_set = list(complete_char_set)
  complete_char_set.sort()
  print("number of unique LaTeX chars in dataset:",len(complete_char_set))
  print("unique LaTeX chars:",complete_char_set)
  return complete_char_set
def get_complete_full_char_set():
  batch_char_set = []
  batch_char_set.append([])
  complete_char_set = set()
  for i in range(1,11):
    input_json_file_name = f"raw_train_data/batch_{i}/JSON/kaggle_data_{i}.json"
    batch_dir = ""
    cur_batch_char_set, cur_batch_char_dict = create_full_char_labels(input_json_file_name,subset_start = 0, subset_end = 5000)
    cur_batch_char_list = list(cur_batch_char_set)
    cur_batch_char_list.sort()
    batch_char_set.append(cur_batch_char_set)
    print(f"batch {i}")
    print(f"number of unique LaTeX chars in batch {i}:{len(cur_batch_char_list)}")
    print(f"unique LaTeX chars",cur_batch_char_list)

  for c in batch_char_set:
    complete_char_set = complete_char_set.union(c)
  complete_char_set = list(complete_char_set)
  complete_char_set.sort()
  print("number of unique LaTeX chars in dataset:",len(complete_char_set))
  print("unique LaTeX chars:",complete_char_set)
  return complete_char_set
def get_char_encoding(complete_char_set):
  char_encoding = {}
  for i, char in enumerate(complete_char_set):
    char_encoding[char] = i
  return char_encoding
def get_reverse_char_encoding(char_encoding):
  reverse_char_encoding = {}
  for key,val in char_encoding.items():
    reverse_char_encoding[val] = key
  return reverse_char_encoding
def encode_char_list(char_list,char_encoding):
  encoded_char_list = []
  for char in char_list:
    encoded_char_list.append(char_encoding[char])
  return encoded_char_list
def decode_char_list(char_list,reverse_char_encoding):
  decoded_char_list = []
  for char in char_list:
    decoded_char_list.append(reverse_char_encoding[char])
  return decoded_char_list
def convert_full_to_visible_encoding(full_char_encoding,visible_char_encoding):
  full_to_visible_encoding = {}
  visible_to_full_encoding = {}
  for key,val in full_char_encoding.items():
    if key in visible_char_encoding.keys():
      full_to_visible_encoding[val + 3] = visible_char_encoding[key] + 3
  full_to_visible_encoding[0] = 0
  full_to_visible_encoding[1] = 1
  full_to_visible_encoding[2] = 2
  for key,val in visible_char_encoding.items():
    if key in full_char_encoding.keys():
      visible_to_full_encoding[val + 3] = full_char_encoding[key] + 3
  visible_to_full_encoding[0] = 0
  visible_to_full_encoding[1] = 1
  visible_to_full_encoding[2] = 2
  return full_to_visible_encoding, visible_to_full_encoding


def apply_nms(orig_prediction, iou_thresh=0.05):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    # print('keep',keep)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

def get_yolo_model(pretrained = False, weight_location = ""):
  if pretrained:
    print("pretrained")
    return YOLO(weight_location)
  print('not pretrained')
  return YOLO('yolov8n.yaml').load('yolov8n.pt')

def get_yolo_prediction(file_path,model):

  prediction = model(file_path,verbose=False)
  prediction = {
      "boxes":prediction[0].boxes.xyxyn,
      "labels":prediction[0].boxes.cls,
      "scores":prediction[0].boxes.conf
  }
  final_prediction = apply_nms(prediction, iou_thresh = 0.5 )
  return final_prediction

def create_sequence_inputs(input_json_file):
    data = []
    with open(f"{input_json_file}", 'r') as f:
        data = json.load(f)
    data = list(data)
    data.sort(key = lambda x: x["uuid"])
    sequence_input_dict = {}
    for d in data:
        # get output file name
        file_name = f"{d['uuid']}.jpg"
        # extract coordinates from each item in json array
        xmins = d["image_data"]["xmins"]
        ymins = d["image_data"]["ymins"]
        xmaxs = d["image_data"]["xmaxs"]
        ymaxs = d["image_data"]["ymaxs"]
        latex_char_labels = encode_char_list(d["image_data"]["visible_latex_chars"],visible_char_encoding)
        # make list of bounding box coordinates for each LaTeX character
        sequence_input_dict[file_name] = [[latex_char,xmin,ymin,xmax,ymax]
                             for latex_char,xmin,ymin,xmax,ymax in zip(latex_char_labels,xmins, ymins, xmaxs, ymaxs)]
    return sequence_input_dict
def create_sequence_outputs(input_json_file):
    data = []
    with open(f"{input_json_file}", 'r') as f:
        data = json.load(f)
    data = list(data)
    data.sort(key = lambda x: x["uuid"])
    sequence_label_dict = {}
    for d in data:
        # get output file name
        file_name = f"{d['uuid']}.jpg"
        # extract coordinates from each item in json array
        latex_char_labels = encode_char_list(d["image_data"]["full_latex_chars"],full_char_encoding)
        # make list of bounding box coordinates for each LaTeX character
        sequence_label_dict[file_name] = latex_char_labels
    return sequence_label_dict


def create_sequence_input(yolo_prediction):
  labels = [int(label) for label in yolo_prediction["labels"].tolist()]
  boxes = yolo_prediction["boxes"].tolist()
  sequence_input = [[label, box[0], box[1], box[2], box[3]]
                    for label, box in zip(labels, boxes)]
  return sequence_input
def sort_inputs_by_position(encoded_inputs):
  encoded_inputs.sort(key = lambda x: (x[-4],x[-3],x[-2],x[-1]))
def add_start_end_tokens(seq,sos_token,eos_token):
  # append one hot encoded sos and eos tokens to start and end of seq
  seq.insert(0,sos_token)
  seq.append(eos_token)
  return seq


class YoloSequenceDataset(Dataset):
    def __init__(self, yolomodel, input_seq_dim = 60, output_seq_dim = 63, starting_index = 0, length = 800):
        # directory of the background images
        self.yolomodel = yolomodel
        self.batch_dir = f"../proc_csv/raw_image"
        self.file_names = sorted([filename
                                  for dirname, _, filenames in os.walk(self.batch_dir)
                                  for i,filename in enumerate(filenames)
                                  if i - starting_index < length
                                  and i >= starting_index])
        print("length file name:", len(self.file_names))
        self.file_paths = [os.path.join(self.batch_dir, file_name) for file_name in self.file_names]
        self.no_of_files = len(self.file_names)
        self.input_seq_dim = input_seq_dim
        self.output_seq_dim = output_seq_dim
        self.object_detection_predictions = {}
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = self.file_paths[idx]
        encoded_labels = []
        encoded_inputs = []
        yolo_prediction = get_yolo_prediction(file_path, self.yolomodel)
        yolo_prediction = create_sequence_input(yolo_prediction)
        encoded_inputs = []
        #create one hot encoding for each label in the source sequence
        for input in yolo_prediction:
          label = input[0]
          # one hot encode each label, starting from i = 2, so we can use i = 0 as sos token and i = 1 as eos token
          one_hot_encoding = [ 1
                              if label + 3 == i and i >= 3
                              else 0
                              for i in range(57)]
          # append bounding box coordinates to the end of the one_hot_encoding to produce input vector of 60 elements
          one_hot_encoding = one_hot_encoding + input[1:5]
          encoded_inputs.append(one_hot_encoding)
        sort_inputs_by_position(encoded_inputs)
        source = encoded_inputs
        item = {
            "source": source
        }
        return item
    def __len__(self):
        return self.no_of_files
    def add_object_detection_prediction(self,filename,object_detection_output):
      self.object_detection_predictions[filename] = object_detection_output