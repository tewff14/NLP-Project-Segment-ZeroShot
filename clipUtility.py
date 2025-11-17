import torch
import clip
import numpy as np
from PIL import Image
import cv2


class CLIPEmbedder:
    """
    A class for embedding text and images using CLIP (Contrastive Language-Image Pre-training).
    """
    
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Initialize the CLIP embedder.
        
        Args:
            model_name: Name of CLIP model to use. Options:
                       - "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"
                       - "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get embedding dimensions
        with torch.no_grad():
            # Get text embedding dimension
            dummy_text = clip.tokenize(["dummy"]).to(self.device)
            dummy_text_features = self.model.encode_text(dummy_text)
            self.text_embedding_dim = dummy_text_features.shape[1]
            
            # Get image embedding dimension
            dummy_image = torch.zeros(1, 3, 224, 224).to(self.device)
            dummy_image_features = self.model.encode_image(dummy_image)
            self.image_embedding_dim = dummy_image_features.shape[1]
    
    def embed_text(self, text, normalize=True):
        """
        Embed text using CLIP.
        
        Args:
            text: String or list of strings to embed
            normalize: If True, normalize embeddings to unit vectors (default: True)
        
        Returns:
            numpy array of text embeddings with shape (n_texts, embedding_dim)
        """
        # Handle single string or list of strings
        if isinstance(text, str):
            text = [text]
        
        # Tokenize text
        text_tokens = clip.tokenize(text).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize if requested
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy and return
        return text_features.cpu().numpy()
    
    def embed_image(self, image, normalize=True, image_format='BGR'):
        """
        Embed image(s) using CLIP.
        
        Args:
            image: Input image(s) as:
                   - numpy array (single image)
                   - PIL Image (single image)
                   - list of numpy arrays or PIL Images (multiple images)
            normalize: If True, normalize embeddings to unit vectors (default: True)
            image_format: Format of input image if numpy array - 'BGR' (default) or 'RGB'
        
        Returns:
            numpy array of image embeddings with shape (n_images, embedding_dim)
        """
        # Handle single image or list of images
        if not isinstance(image, list):
            image = [image]
        
        # Preprocess images
        processed_images = []
        for img in image:
            # Convert to PIL Image if needed
            if isinstance(img, np.ndarray):
                # Convert BGR to RGB if needed
                if image_format.upper() == 'BGR' and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Preprocess using CLIP's preprocessing
            processed_img = self.preprocess(img)
            processed_images.append(processed_img)
        
        # Stack images into batch tensor
        image_tensor = torch.stack(processed_images).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            
            # Normalize if requested
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy and return
        return image_features.cpu().numpy()
    
    def compute_similarity(self, text_embeddings, image_embeddings):
        """
        Compute cosine similarity between text and image embeddings.
        
        Args:
            text_embeddings: Text embeddings array with shape (n_texts, embedding_dim)
            image_embeddings: Image embeddings array with shape (n_images, embedding_dim)
        
        Returns:
            Similarity matrix with shape (n_texts, n_images) where each value is cosine similarity
        """
        # Convert to tensors if needed
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.from_numpy(text_embeddings).to(self.device)
        if isinstance(image_embeddings, np.ndarray):
            image_embeddings = torch.from_numpy(image_embeddings).to(self.device)
        
        # Compute cosine similarity (dot product if normalized)
        similarity = (100.0 * text_embeddings @ image_embeddings.T).softmax(dim=-1)
        
        return similarity.cpu().numpy()
    


