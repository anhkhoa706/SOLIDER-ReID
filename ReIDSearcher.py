import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from re_ranking_feature import re_ranking


class ReIDSearcher:
    def __init__(self, config_file, weight_path, re_ranking=True, device=None):
        self.config_file = config_file
        self.weight_path = weight_path
        self.re_ranking = re_ranking
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = None
        self.model = None
        self.img_emb_dict = {}
        self.setup_environment()
        self.load_model()

    def setup_environment(self):
        if self.config_file:
            cfg.merge_from_file(self.config_file)
        cfg.TEST.WEIGHT = self.weight_path
        cfg.TEST.RE_RANKING = self.re_ranking
        # cfg.freeze()

        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.logger = setup_logger("transreid", output_dir, if_train=False)
        self.logger.info("Configuration Loaded")
        if self.config_file:
            with open(self.config_file, 'r') as cf:
                self.logger.info(f"Loaded configuration:\n{cf.read()}")
        self.logger.info(f"Running with config:\n{cfg}")

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    def load_model(self):
        train_loader, _, _, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
        self.model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                                view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)

        if cfg.TEST.WEIGHT:
            self.logger.info(f"Loading model weights from {cfg.TEST.WEIGHT}")
            self.model.load_param(cfg.TEST.WEIGHT)

        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    def extract_features(self, image_tensor):
        with torch.no_grad():
            features = self.model(image_tensor)
            if isinstance(features, tuple):
                features = features[0]
        return features.cpu().numpy()

    @staticmethod
    def cosine_similarity(feature1, feature2):
        feature1 = torch.tensor(feature1)
        feature2 = torch.tensor(feature2)
        return F.cosine_similarity(feature1, feature2).item()

    def search_top_k_matches(self, query_img_path, images_folder, k=5):
        query_tensor = self.preprocess_image(query_img_path).to(self.device)
        query_features = self.extract_features(query_tensor)
        
        results = []
        for img_name in tqdm(os.listdir(images_folder)):
            img_path = os.path.join(images_folder, img_name)
            candidate_tensor = self.preprocess_image(img_path).to(self.device)
            candidate_features = self.extract_features(candidate_tensor)
            self.img_emb_dict[img_name] = candidate_features
            score = self.cosine_similarity(query_features, candidate_features)
            results.append((img_name, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top_k_matches = results[:k]
        
        if self.re_ranking:
            print("*With ReRanking")
            top_k_matches = self.re_rank_results(query_features, top_k_matches, k)
        
        return top_k_matches

    def re_rank_results(self, query_features, top_k_matches, k=5):
        emb_k = [self.img_emb_dict[name] for name, _ in top_k_matches]
        emb_k = np.array(emb_k).reshape(k, -1)
        query_emb = np.array(query_features).reshape(1, -1)
        
        # Set parameters
        k1, k2, lambda_value = 20, 6, 0.3
        # Perform re-ranking
        final_dist = re_ranking(query_emb, emb_k, k1, k2, lambda_value)
        
        # Compute top-k gallery indices for each query
        top_k_indices = np.argsort(final_dist, axis=1)[0][:k]

        reranked_matches = []
        for ind in top_k_indices:
            # Get value
            emb_in_final = np.array(emb_k[ind])
            score = final_dist[0][ind]
            for name, emb in self.img_emb_dict.items():
                emb = np.array(emb)
                # Flatten Embedding
                # Compare two numby arrays
                if np.array_equal(emb_in_final.flatten(), emb.flatten()):
                    reranked_matches.append((name, score))
        return reranked_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Inference and Image Search")
    parser.add_argument("--config_file", default="configs/msmt17/swin_base.yml", help="Path to config file", type=str)
    parser.add_argument("--query_image", help="Path to the query image", type=str, required=True, default="datasets/market1501/7/0173_c04s1_000696_02_t.jpg")
    parser.add_argument("--image_folder", help="Path to the folder containing candidate images", type=str, required=True, default="datasets/market1501/7")
    parser.add_argument("--top_k", help="Number of top matches to return", type=int, default=5)

    args = parser.parse_args()

    searcher = ReIDSearcher(config_file=args.config_file, weight_path='weights/swin_base_msmt17.pth')
    top_matches = searcher.search_top_k_matches(args.query_image, args.image_folder, k=args.top_k+1)
    print("_________________________Config__________________________")
    print(f"Config_file: {args.config_file}")
    print(f"Query: {args.query_image}")
    print(f"Images folder: {args.image_folder}")
    print("_________________________________________________________")
    
    print(f"Top {args.top_k} Matches")
    for rank, (img_name, score) in enumerate(top_matches, start=1):
        if rank != 1: 
            print(f"Rank {rank-1}: {img_name}, Score: {score:.4f}")
