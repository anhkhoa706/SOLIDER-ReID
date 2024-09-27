import os
import torch
from config import cfg
from datasets import make_dataloader
from model import make_model
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import argparse
from utils.logger import setup_logger
from tqdm import tqdm

WEIGHT_PATH = 'weights/swin_small_market.pth'

# Set up the environment and logger
def setup_environment(args):
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.TEST.WEIGHT = WEIGHT_PATH
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Running with config:\n{cfg}")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    return logger

# Load the ReID model from the .pth checkpoint
def load_model():
    # Load data loader to get number of classes, cameras, views, etc.
    train_loader, _, _, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    # Create model based on config
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    
    # Load pre-trained weights if specified in the config
    if cfg.TEST.WEIGHT != '':
        print(f"Loading model weights from {cfg.TEST.WEIGHT}")
        model.load_param(cfg.TEST.WEIGHT)
    
    # Move the model to GPU or CPU based on availability
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    return model

# Preprocess the images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Extract features from a given image
def extract_features(model, image_tensor):
    with torch.no_grad():
            features = model(image_tensor)
            if isinstance(features, tuple):  # If the model output is a tuple
                features = features[0]
    return features.cpu().numpy()

# Calculate cosine similarity
def cosine_similarity(feature1, feature2):
    feature1 = torch.tensor(feature1)
    feature2 = torch.tensor(feature2)
    return F.cosine_similarity(feature1, feature2).item()

# Search for the best match in a folder
def search_best_match(model, query_img_path, images_folder):
    query_tensor = preprocess_image(query_img_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    query_features = extract_features(model, query_tensor)
    
    best_match = None
    best_score = -1
    
    for img_name in tqdm(os.listdir(images_folder)):
        img_path = os.path.join(images_folder, img_name)
        candidate_tensor = preprocess_image(img_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        candidate_features = extract_features(model, candidate_tensor)
        
        score = cosine_similarity(query_features, candidate_features)
        
        if score > best_score:
            best_score = score
            best_match = img_name
    
    return best_match, best_score

def search_top_k_matches(model, query_img_path, images_folder, k=5):
    query_tensor = preprocess_image(query_img_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    query_features = extract_features(model, query_tensor)
    
    results = []  # List to store tuples of (image_name, score)
    
    for img_name in tqdm(os.listdir(images_folder)):
        img_path = os.path.join(images_folder, img_name)
        candidate_tensor = preprocess_image(img_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        candidate_features = extract_features(model, candidate_tensor)
        
        score = cosine_similarity(query_features, candidate_features)
        
        results.append((img_name, score))
    
    # Sort results by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k matches
    top_k_matches = results[:k]
    return top_k_matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Inference and Image Search")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--query_image", help="Path to the query image", type=str, required=True)
    parser.add_argument("--image_folder", help="Path to the folder containing candidate images", type=str, required=True)

    args = parser.parse_args()

    logger = setup_environment(args)
    
    # Load the model
    model = load_model()
    
    # Search for the top 5 matches
    top_matches = search_top_k_matches(model, args.query_image, args.image_folder, k=5)
    
    print("Top 5 Matches:")
    for i, (img_name, score) in enumerate(top_matches):
        print(f"Rank {i+1}: {img_name}, Similarity Score: {score}")
