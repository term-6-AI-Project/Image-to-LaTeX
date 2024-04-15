# AI-Project
To run ai_project_final.ipynb on google colab, drop the 
1. "project-50021-415714-5d993bed20f3.json" (google drive service account credentials)
2. "kaggle.json" (kaggle credentials)
files to the same directory as the jupyter notebook. Then proceed to run all cells in the notebook. 
**Warning: when prompted to "restart kernel" after the line `!pip install yolov8`, click "Cancel" as you don't need to restart kernel.** 
## YOLO v8 fine-tuned model
The notebook will load the fine-tuned YOLO model params from our google drive with title:
"yolo_params_batch_1234567_epochs_12"

To fine-tune the YOLO v8 model, uncomment section titled "Part 2.1.1 Train YOLO Model"
## LSTM With attention params
and the LSTM model will load the best model params from our google drive with title: 
"c_lstm_1_layer_512_with_attention_and_params_batch_1234567_epoch_5". 

To train the LSTM with attention model from scratch, replace:
`lstm_model = get_lstm_model(weight_location = "c_lstm_1_layer_512_with_attention_and_params_batch_1234567_epoch_5")`
with
`lstm_model = get_lstm_model(weight_location = "")`

## Running the GUI
Installing dependencies:
streamlit:
!pip install streamlit
YOLO:
!pip install yolov8

navigate to the project GUI directory and run the app using the command:
streamlit run Project.py
