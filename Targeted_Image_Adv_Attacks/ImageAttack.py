from typing import Callable, Optional, Tuple
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
import torchattacks
from skimage.metrics import structural_similarity as ssim

PREPROCESSINGS = {
    'Res256Crop224': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'Crop288': transforms.Compose([
        transforms.CenterCrop(288),
        transforms.ToTensor()
    ]),
    None: transforms.Compose([
        transforms.ToTensor()
    ]),
}

# return index and file path which is quite useful when reform the image as adversarial examples
class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label, index, self.imgs[index][0]  
        

def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=True,  # keep shuffle=True
                                  num_workers=0)

    x_test, y_test, indices, file_paths = [], [], [], []
    for i, (x, y, idx, path) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        indices.extend(idx.tolist())
        file_paths.extend(path)
        if n_examples is not None and len(x_test) * batch_size >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]
        indices = indices[:n_examples]
        file_paths = file_paths[:n_examples]

    return x_test_tensor, y_test_tensor, indices, file_paths

def load_advdata(
    n_examples: Optional[int] = None,
    data_dir: str = './adv_image',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
    dataset = IndexedImageFolder(root=data_dir,
                                 transform=transforms_test)
    return _load_dataset(dataset, n_examples)

# Load all images without limiting the number
images, labels, original_indices, file_paths = load_advdata()

# create a mapping from the original index to the new index
index_mapping = {idx: i for i, idx in enumerate(original_indices)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    PATH = 'ckpt_densenet121_catdogfox_classify.pth'
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 3) 
    )
    model.load_state_dict(torch.load(PATH, map_location=device))
    return model.to(device)

def setup_attack(model):
    atk = torchattacks.PGD(model, eps=4/255, alpha=0.5/255, steps=12, random_start=True)
    atk.set_mode_targeted_by_label()
    return atk

def save_adversarial_image(adv_image, original_index, file_path):
    new_index = index_mapping[original_index]
    category = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    save_path = os.path.join(f'update/{category}', filename)
    adv_image.save(save_path)

def process_batch(start_idx, end_idx, images, labels, atk, model, target_mapping, to_pil, original_indices, file_paths):
    batch_images = images[start_idx:end_idx].to(device)
    batch_labels = labels[start_idx:end_idx].to(device)
    
    batch_new_labels = batch_labels.clone()
    for original_label, target_label in target_mapping.items():
        batch_new_labels[batch_labels == original_label] = target_label
    
    batch_adv_images = atk(batch_images, batch_new_labels)
    adv_pred = model(batch_adv_images)
    print("Original:", batch_labels)
    print("Target  :", batch_new_labels)
    print("Attacked:", torch.argmax(adv_pred, 1))

    attack_success = (torch.argmax(adv_pred, 1) == batch_new_labels).float().mean().item()
    print(f"Attack Success Rate: {attack_success:.4f}")

    batch_ssim_values = []
    
    for i in range(batch_adv_images.shape[0]):
        img_idx = start_idx + i
        original_index = original_indices[img_idx]
        file_path = file_paths[img_idx]
        
        adv_img_tensor = batch_adv_images[i].cpu()
        orig_img_tensor = batch_images[i].cpu()
        
        adv_img_tensor = adv_img_tensor.clamp(0, 1) * 255
        orig_img_tensor = orig_img_tensor * 255
        
        adv_img_np = adv_img_tensor.byte().permute(1, 2, 0).numpy()
        orig_img_np = orig_img_tensor.byte().permute(1, 2, 0).numpy()
        
        ssim_value = ssim(orig_img_np, adv_img_np, channel_axis=2, data_range=255)
        batch_ssim_values.append(ssim_value)
        
        adv_img = to_pil(adv_img_tensor.byte())
        save_adversarial_image(adv_img, original_index, file_path)
    
    print(f"Processed images {start_idx} to {end_idx}")
    print(f"Batch SSIM values: {batch_ssim_values}")
    print(f"Average SSIM for this batch: {np.mean(batch_ssim_values):.4f}")
    print()

    return np.mean(batch_ssim_values), attack_success

def attack_category(start_index, images, labels, original_indices, file_paths):
    print(f"Starting attack on images from index {start_index}")
    model = load_model()
    print('[Model loaded]')
    atk = setup_attack(model)
    
    to_pil = transforms.ToPILImage()
    os.makedirs('update/cat', exist_ok=True)
    os.makedirs('update/dog', exist_ok=True)
    os.makedirs('update/fox', exist_ok=True)
    
    batch_size = 10
    total_images = 50
    ssim_values = []
    success_rates = []
    
    for start_idx in range(start_index, start_index + total_images, batch_size):
        end_idx = min(start_idx + batch_size, start_index + total_images)
        ssim_value, success_rate = process_batch(start_idx, end_idx, images, labels, atk, model, target_mapping, to_pil, original_indices, file_paths)
        ssim_values.append(ssim_value)
        success_rates.append(success_rate)
    
    print(f"Average SSIM for this batch: {np.mean(ssim_values):.4f}")
    print(f"Average Attack Success Rate for this batch: {np.mean(success_rates):.4f}")
    print(f"Attack on images completed")
    print()

    return np.mean(ssim_values), np.mean(success_rates)

# main
target_mapping = {0: 1, 1: 2, 2: 0}

cat_ssim, cat_success = attack_category(0, images, labels, original_indices, file_paths)
dog_ssim, dog_success = attack_category(50, images, labels, original_indices, file_paths)
fox_ssim, fox_success = attack_category(100, images, labels, original_indices, file_paths)

print("All attacks completed.")
print(f"Overall Average SSIM: {np.mean([cat_ssim, dog_ssim, fox_ssim]):.4f}")
print(f"Overall Average Attack Success Rate: {np.mean([cat_success, dog_success, fox_success]):.4f}")