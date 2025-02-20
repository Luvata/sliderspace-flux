import torch
import clip
import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate CLIP embeddings and PCA')
    parser.add_argument('image_dir', type=str, help='Directory containing images')
    parser.add_argument('--embed_output_dir', type=str, default='embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--pca_output_file', type=str, default='components',
                       help='Where to save PCA components')
    parser.add_argument('--bs', type=int, default=32,
                       help='Batch size for processing')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.embed_output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_files = [f for f in os.listdir(args.image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(image_files), args.bs)):
        batch_files = image_files[i:i + args.bs]
        batch_images = []
        
        for img_file in batch_files:
            image_path = os.path.join(args.image_dir, img_file)
            image = preprocess(Image.open(image_path)).unsqueeze(0)
            batch_images.append(image)
        
        batch_tensor = torch.cat(batch_images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)
            image_features = image_features.cpu().numpy()
        
        for j, img_file in enumerate(batch_files):
            embedding = image_features[j]
            output_path = os.path.join(args.embed_output_dir, 
                                     os.path.splitext(img_file)[0] + '.npy')
            np.save(output_path, embedding)
            all_embeddings.append(embedding)
    
    all_embeddings = np.stack(all_embeddings)
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    pca = PCA()
    pca.fit(all_embeddings)
    np.save(os.path.join(args.pca_output_file), pca.components_)

if __name__ == "__main__":
    main()