import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import json
import time
import argparse
from torchvision import transforms

def image_preprocess(image_path):
    """
    Preprocesses the image for input into the model.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - image (np.ndarray): The preprocessed image as numpy array
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).numpy()
    return np.expand_dims(image, axis=0)  # Add batch dimension

def test_onnx(model_path, data_path, device_type='gpu', device_id=0):
    """
    Tests the ONNX model on the given dataset and calculates relative errors.

    Parameters:
    - model_path (str): The path to the ONNX model file.
    - data_path (str): The path to the dataset directory.
    - device_type (str): The device to use for computation ('gpu' or 'cpu')
    - device_id (int): The device index to use for GPU
    """
    # Configure execution providers
    providers = ['CUDAExecutionProvider'] if device_type == 'gpu' else ['CPUExecutionProvider']
    provider_options = [{'device_id': device_id}] if device_type == 'gpu' else None
    
    # Create ONNX session
    session = ort.InferenceSession(
        model_path, 
        providers=providers,
        provider_options=provider_options
    )
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Using execution providers: {session.get_providers()}")
    print(f"Running on {'GPU' if device_type == 'gpu' else 'CPU'}")

    # Initialize metrics
    preprocess_image_time = []
    model_time = []
    RE1 = []
    RE2 = []
    categories = ['High', 'Low', 'Mid']
    
    # Load ground truth data
    with open('label.json', 'r', encoding='utf-8') as file:
        GT = json.load(file)

    print("Starting prediction")
    start_time = time.time()
    
    # Warm-up run
    warmup_image = os.path.join(data_path, 'High/03-58-2.png')
    input_data = image_preprocess(warmup_image)
    session.run([output_name], {input_name: input_data})
    
    for category in categories:
        category_dir = os.path.join(data_path, category)
        for filename in os.listdir(category_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(category_dir, filename)
                preprocess_image_start = time.time()
                input_data = image_preprocess(image_path)
                preprocess_image_time.append(time.time() - preprocess_image_start)
                
                
                # Inference
                inference_start = time.time()
                pred_logits = session.run([output_name], {input_name: input_data})[0]
                model_time.append(time.time() - inference_start)
                
                # Process outputs
                pred_softmax = np.exp(pred_logits) / np.sum(np.exp(pred_logits), axis=1, keepdims=True)
                top_n = np.argsort(pred_softmax[0])[::-1][:3]
                confs = pred_softmax[0][top_n]
                
                # Calculate flow values
                flow1 = 0.0
                for i in range(3):
                    class_name = categories[top_n[i]]
                    if class_name == 'High':
                        flow1 += 9.719 * confs[i]
                        if i == 0:
                            flow2 = 9.719
                    elif class_name == "Low":
                        flow1 += 3.0 * confs[i]
                        if i == 0:
                            flow2 = 3.0
                    elif class_name == 'Mid':
                        flow1 += 7.5 * confs[i]
                        if i == 0:
                            flow2 = 7.5
                
                # Calculate errors
                GT_flow = GT[os.path.basename(image_path)]['flow']
                if 3 < GT_flow < 12:
                    error1 = abs(flow1 - GT_flow) / GT_flow
                    error2 = abs(flow2 - GT_flow) / GT_flow
                    RE1.append(error1)
                    RE2.append(error2)
    
    # Calculate and print results
    end_time = time.time()
    avg_re1 = sum(RE1) / len(RE1) if RE1 else 0
    avg_re2 = sum(RE2) / len(RE2) if RE2 else 0
    
    print("\nTest results:")
    print(f"Total samples processed: {len(model_time)}")
    print(f"Average weighted flow RE: {avg_re1:.2%}")
    print(f"Average classification flow RE: {avg_re2:.2%}")
    print(f"Total execution time: {end_time - start_time:.2f}s")
    print(f"Preprocess time breakdown:")
    print(f"  - Total preprocess time: {sum(preprocess_image_time):.4f}s")
    print(f"  - Average per sample: {sum(preprocess_image_time)/len(preprocess_image_time)*1000:.2f}ms")
    print(f"  - FPS: {len(preprocess_image_time)/sum(preprocess_image_time):.2f}")
    print(f"Inference time breakdown:")
    print(f"  - Total inference time: {sum(model_time):.4f}s")
    print(f"  - Average per sample: {sum(model_time)/len(model_time)*1000:.2f}ms")
    print(f"  - FPS: {len(model_time)/sum(model_time):.2f}")
    print(f"Total time per sample: {(sum(preprocess_image_time) + sum(model_time)) / len(model_time) * 1000:.2f}ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ONNX model performance')
    parser.add_argument('--model_path', type=str, default="./onnx_models/model.onnx", help='Path to ONNX model')
    parser.add_argument('--data_path', type=str, default="./datasets/Split_data/test", help='Path to test dataset')
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'], help='Device to use for inference')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID (0 by default)')
    args = parser.parse_args()
    
    test_onnx(
        args.model_path,
        args.data_path,
        device_type=args.device,
        device_id=args.device_id
    )