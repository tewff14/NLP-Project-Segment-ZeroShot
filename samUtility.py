import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry


class SAMSegmenter:
    """
    A class for segmenting images using SAM (Segment Anything Model) with bounding boxes.
    """
    
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth"):
        """
        Initialize the SAM segmenter.
        
        Args:
            model_type: Type of SAM model - "vit_h", "vit_l", or "vit_b"
            checkpoint_path: Path to the SAM model checkpoint file
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = None
        self.current_image = None
    
    def segment_withbox(self, image, boxes, image_format='BGR'):
        """
        Segment an image using SAM with a list of bounding boxes.
        
        Args:
            image: Input image as numpy array
            boxes: List of bounding boxes, where each box is [x1, y1, x2, y2]
            image_format: Format of input image - 'BGR' (default, for cv2.imread) or 'RGB'
        
        Returns:
            List of tuples containing (masks, scores, logits) for each box
        """
        # Convert image to RGB format (SAM expects RGB)
        if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
            if image_format.upper() == 'BGR':
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image  # Already RGB
        else:
            image_rgb = image
        
        # Create predictor if not exists or if image changed
        if self.predictor is None or not np.array_equal(self.current_image, image_rgb):
            self.predictor = SamPredictor(self.sam)
            self.predictor.set_image(image_rgb)
            self.current_image = image_rgb
        
        results = []
        for box in boxes:
            box_array = np.array(box)
            masks, scores, logits = self.predictor.predict(
                box=box_array,
                multimask_output=True
            )
            results.append((masks, scores, logits))
        
        return results
    
    def get_masked_image(self, image, masks, mask_index=0, image_format='BGR', 
                         output_format='BGR', overlay=False, alpha=0.5, 
                         background_color=(0, 0, 0)):
        """
        Apply masks to an image and return the masked image.
        
        Args:
            image: Input image as numpy array
            masks: Mask array from SAM prediction (can be single mask or array of masks)
                   If array of masks, mask_index determines which mask to use
            mask_index: Index of mask to use if masks is an array (default: 0, best mask)
            image_format: Format of input image - 'BGR' (default) or 'RGB'
            output_format: Format of output image - 'BGR' (default) or 'RGB'
            overlay: If True, overlay mask on original image. If False, show only masked regions
            alpha: Transparency for overlay mode (0.0 to 1.0, default: 0.5)
            background_color: Background color for non-masked regions in non-overlay mode 
                            (tuple in same format as image_format, default: (0, 0, 0) for black)
        
        Returns:
            Masked image as numpy array
        """
        # Convert image to RGB for processing
        if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
            if image_format.upper() == 'BGR':
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image.copy()
        else:
            image_rgb = image.copy()
        
        # Extract the mask (handle both single mask and array of masks)
        if len(masks.shape) == 3:
            # Array of masks, select the one at mask_index
            mask = masks[mask_index]
        else:
            # Single mask
            mask = masks
        
        # Ensure mask is boolean
        mask_bool = mask.astype(bool)
        
        if overlay:
            # Overlay mode: blend mask with original image
            # Create colored mask overlay
            mask_colored = np.zeros_like(image_rgb)
            mask_colored[mask_bool] = [255, 0, 0]  # Red overlay
            
            # Blend with original image
            masked_image = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)
        else:
            # Non-overlay mode: show only masked regions, rest is background
            masked_image = image_rgb.copy()
            # Convert background_color from BGR to RGB if needed
            if image_format.upper() == 'BGR':
                # background_color is in BGR, convert to RGB
                bg_color_rgb = (background_color[2], background_color[1], background_color[0])
            else:
                # background_color is already in RGB
                bg_color_rgb = background_color
            # Set non-masked regions to background color
            masked_image[~mask_bool] = bg_color_rgb
        
        # Convert output format if needed
        # masked_image is currently in RGB format
        if output_format.upper() == 'BGR':
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        # If output_format is RGB, masked_image is already in RGB
        
        return masked_image
    
    def get_masked_images_from_results(self, image, results, mask_index=0, 
                                       image_format='BGR', output_format='BGR',
                                       overlay=False, alpha=0.5, 
                                       background_color=(0, 0, 0)):
        """
        Apply masks from segment_withbox results to an image.
        
        Args:
            image: Input image as numpy array
            results: Results from segment_withbox method (list of (masks, scores, logits) tuples)
            mask_index: Index of mask to use from each result (default: 0, best mask)
            image_format: Format of input image - 'BGR' (default) or 'RGB'
            output_format: Format of output image - 'BGR' (default) or 'RGB'
            overlay: If True, overlay all masks on original image. If False, show only masked regions
            alpha: Transparency for overlay mode (0.0 to 1.0, default: 0.5)
            background_color: Background color for non-masked regions in non-overlay mode 
                            (tuple in same format as image_format, default: (0, 0, 0) for black)
        
        Returns:
            List of masked images, one for each result
        """
        masked_images = []
        for masks, scores, logits in results:
            masked_img = self.get_masked_image(
                image, masks, mask_index, image_format, 
                output_format, overlay, alpha, background_color
            )
            masked_images.append(masked_img)
        
        return masked_images