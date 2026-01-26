import torch.nn as nn
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import os
import json
from time import time
from math import pow
from models.LAMNet import LAMNet
import argparse
import numpy as np
import datetime
import csv

def image_preprocess(image_path,device):
    """
    Preprocesses the image for input into the model.

    Parameters:
    - image_path (str): The path to the image file.
    - device (str): The device to which the image tensor should be moved ('cuda' or 'cpu').

    Returns:
    - image (torch.Tensor): The preprocessed image tensor.
    """

    # just resize and normalize, no augmentation, data augmentation only for training
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    return image

def test(model_path, data_path, using_device, model_id, flow_range=[3,12], model_name="lamnet"):
    """
    Tests the model on the given dataset and calculates relative errors.

    Parameters:
    - model_path (str): The path to the model file.
    - data_path (str): The path to the dataset directory.
    - using_device (str): The device to use for computation ('idx for cuda' or 'cpu').
    """
    model_time = []
    # add model_name to the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f'F:\\server_file\\zsh_datasets\\splits_1s\\flowprediction\\flow_results_csv\\{model_name}_results.csv'
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model_ID', 'Flow_Min', 'Flow_Max', 
                'AE_weighting', 'AE_classification', 
                'RE_weighting', 'RE_classification',
                'Total_Time', 'Avg_Inference_Time',
                'Num_Samples'
            ])
    RE1 = []
    RE2 = []
    AE1 = []
    AE2 = []
    categories = ['High', 'Low', 'Medium']
    with open('F:\\server_file\\zsh_datasets\\splits_1s\\flowprediction\\label2.json','r',encoding='utf-8') as file:
        GT = json.load(file)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{using_device}')
        print("GPU is available, using:",device)
    else:
        print('No GPU available, using CPU.')
        device = torch.device('cpu')

    model = LAMNet(num_classes=3)


    print("Loading model: ", model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    print('Start prediction')
    start_time = time()
    for category in categories:
        category_dir = os.path.join(data_path,category)
        for filename in os.listdir(category_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(category_dir,filename)
                image = image_preprocess(image_path, device)
                with torch.no_grad():
                    t0 = time()
                    pred_logits = model(image)
                    model_time.append(time()-t0)
                    pred_softmax = F.softmax(pred_logits, dim=1)
                    top_n = torch.topk(pred_softmax, 3)
                    pred_ids = top_n[1].cpu().detach().numpy().squeeze()
                    pred_ids = pred_ids.tolist()
                    confs = top_n[0].cpu().detach().numpy().squeeze()
                    flow1 = 0
                    for i in range(3):
                        class_name = categories[pred_ids[i]]
                        confidence = confs[i] * 100
                        if class_name == 'High':
                            # flow1 += ((10.438-9)/2 +9)*confs[i]
                            flow1 += 9.719 * confs[i]
                            if i==0:
                                # flow2 = (10.438-9)/2 +9
                                flow2 = 9.719*confs[i]
                        elif class_name == "Low":
                            flow1 += 3*confs[i]
                            if i==0:
                                flow2 = 3
                        elif class_name == 'Medium':
                            flow1 += 7.5*confs[i]
                            if i==0:
                                flow2 = 7.5
                    GT_flow = float(GT[os.path.basename(image_path)[17:22]]['flow'])
                    if GT_flow > flow_range[0] and GT_flow < flow_range[1]:
                        error1 = abs(flow1 - GT_flow) / GT_flow
                        error2 = abs(flow2 - GT_flow) / GT_flow

                        AE1.append(abs(flow1 - GT_flow))
                        AE2.append(abs(flow2 - GT_flow))
                        RE1.append(error1)
                        RE2.append(error2)
    end_time = time()

    # Calculate the averages
    avg_ae1 = sum(AE1)/len(AE1) if AE1 else 0
    avg_ae2 = sum(AE2)/len(AE2) if AE2 else 0
    avg_re1 = sum(RE1)/len(RE1) if RE1 else 0
    avg_re2 = sum(RE2)/len(RE2) if RE2 else 0
    total_time = end_time - start_time
    avg_inference = sum(model_time)/len(model_time) if model_time else 0
    num_samples = len(AE1)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            model_id,
            flow_range[0],
            flow_range[1],
            f"{avg_ae1:.6f}",
            f"{avg_ae2:.6f}",
            f"{avg_re1:.6%}".strip('%'),  
            f"{avg_re2:.6%}".strip('%'),
            f"{total_time:.2f}",
            f"{avg_inference:.4f}",
            num_samples
        ])
    
    print(f"\nModel {model_id} | Flow range: {flow_range}")
    print(f"Avg AE (weighting): {avg_ae1:.4f}")
    print(f"Avg AE (classification): {avg_ae2:.4f}")
    print(f"Avg RE (weighting): {avg_re1:.4%}")
    print(f"Avg RE (classification): {avg_re2:.4%}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg inference time: {avg_inference:.4f}s")
    print(f"Processed samples: {num_samples}")
    print('-' * 50)
    print('Done!')
    print('Running time: {:.2f}s'.format(end_time-start_time))

    print(f"Total Inference time: {sum(model_time)}")
    print(f"Average inference time: {sum(model_time)/len(model_time)}")

if __name__== '__main__':

    the_device = "0"
    flow_range = [[3,6],[3,9],[3,12],[6,9],[6,12],[9,12]]
    model_ids = [1,2,3,4,5]
    total_start = time()
    
    for flow in flow_range:
        for model_id in model_ids:
            print(f"\n{'='*60}")
            print(f"Testing model {model_id} with flow range {flow}")
            print(f"{'='*60}")
            
            test(
                model_path=f"F:\\server_file\\zsh_datasets\\splits_1s\\flowprediction\\new_flow_dataset_runs\\KFOLD_2025_1129_1536/fold_{model_id}/models/best_model_fold{model_id}.pth", 
                data_path="F:\\server_file\\zsh_datasets\\splits_1s\\flowprediction\\datasets\\new_flow_prediction_dataset\\test", 
                using_device=the_device,
                model_id=model_id,
                flow_range=flow,
                model_name="LAMNet"
            )
    
    total_time = time() - total_start
    print(f"\n{'#'*60}")
    print(f"All tests completed! Total time: {total_time/60:.2f} minutes")
    print(f"Results saved to CSV with timestamp")
    print(f"{'#'*60}")


