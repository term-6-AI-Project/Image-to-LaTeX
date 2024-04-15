import streamlit as st
import time
import os
# from model import prediction
from model import yolo
import torch
from torch.utils.data import DataLoader
from model import model
from model import prediction

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()

st.set_page_config(
    page_title = 'Image to LaTex',
)

st.title('Image to LaTex Project')
st.header('Converting your handwritten equation to LaTeX')
st.subheader('50.021 Artificial Intelligence')
st.divider()

def load_image(image_file):
    img = open(image_file)
    return img

def add_start_end_tokens(seq,sos_token,eos_token):
  seq.insert(0,sos_token)
  seq.append(eos_token)
  return seq

def pad_seq(batch_seq, max_seq_length, pad_token, sos_token,eos_token):
  new_batch_seq = []
  for seq in batch_seq:
    new_seq = seq
    while len(new_seq) < max_seq_length:
      new_seq.append(pad_token)
    new_seq = add_start_end_tokens(new_seq, sos_token,eos_token)
    new_batch_seq.append(torch.tensor(new_seq))
  return new_batch_seq

def collate_fn(batch):
  src_pad_token = [1 if i == 0
                    else 0
                    for i in range(61)]
  src_sos_token = [1 if i == 1
                    else 0
                    for i in range(61)]
  src_eos_token = [1 if i == 2
                    else 0
                    for i in range(61)]
  batch_src = [item["source"] for item in batch]
  max_src_length = len(max(batch_src, key = lambda t: len(t)))
  batch_src = pad_seq(batch_src, max_src_length, src_pad_token, src_sos_token, src_eos_token)
  batch_src = torch.stack(batch_src,dim=0)
  return batch_src

class howto():
    st.header('How to?')
    st.markdown('''
    How to use Converter:
    1. Upload an image containing handwritten math equations
    2. Wait for us to process your images
    3. Once processing is done, we will return you the equation in LaTeX!
    
    '''            )
    st.divider()

class main():
    st.header("Please upload your image here")
    image_file = st.file_uploader('Upload', type=['jpg', 'png', 'jpeg', 'pdf'])
    
    def save_uploaded_file(uploaded_file):
    # Define the directory where you want to save the uploaded file
        save_dir = "../proc_csv/raw_image"
        os.makedirs(save_dir, exist_ok=True)
        files = os.listdir(save_dir)
        if files:
        # If there's a file, delete it
            os.remove(os.path.join(save_dir, files[0]))
        # Save the uploaded file with a unique name
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    if image_file is not None:
        with st.spinner("Uploading..."):
            time.sleep(3)
            file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": image_file.size}
            st.write(file_details)
            image_bytes = image_file.read()
            
            save_uploaded_file(image_file) #save file here
            
            st.image(image_bytes)

        if st.button('Process Image Now', on_click=None, type='primary'):
            with st.spinner('Processing Image...'):
                time.sleep(5)
                st.write("Image Processed Successfully")
                image_path = '../proc_csv/raw_image/{}'.format(image_file.name)       
                
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "yolo_params_batch_1234567_epochs_12.pt")

                yolo_model = yolo.get_yolo_model(pretrained=True, weight_location=model_path)
                
                yolo_dataset = yolo.YoloSequenceDataset(yolo_model)
                
                yolo_dataloader = DataLoader(yolo_dataset, batch_size=1, collate_fn=collate_fn)
                
                lstm_model = model.get_LSTM_model()

                output_equation = prediction.test_fn(lstm_model, yolo_dataloader)

                st.markdown(f"Your LaTex Equation is: {output_equation}")

                        

