import faiss
import numpy as np


class Faiss:
    """
    A wrapper class for Faiss similarity search and clustering.
    Supports both CPU and GPU operations, multiple index types, and various distance metrics.
    """
    
    def __init__(self, dim, index_type='L2', use_gpu=False, gpu_id=0):
        """
        Initialize the Faiss index.
        
        Args:
            dim: Dimension of the vectors
            index_type: Type of index to use:
                       - 'L2': L2 (Euclidean) distance (default)
                       - 'IP': Inner product (for cosine similarity with normalized vectors)
            use_gpu: Whether to use GPU acceleration (default: False)
            gpu_id: GPU device ID to use if use_gpu is True (default: 0)
        """
        self.dim = dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Create appropriate index based on type
        if index_type.upper() == 'L2':
            self.index = faiss.IndexFlatL2(dim)
        elif index_type.upper() == 'IP':
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}. Use 'L2' or 'IP'")
        
        # Move to GPU if requested
        if use_gpu:
            if not faiss.get_num_gpus():
                raise RuntimeError("No GPU available. Set use_gpu=False or install GPU-enabled Faiss.")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
            self.gpu_res = res
        else:
            self.gpu_res = None
    
    def add(self, data):
        """
        Add vectors to the index.
        
        Args:
            data: numpy array of shape (n_vectors, dim) with dtype float32
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if len(data.shape) != 2:
            raise ValueError(f"Data must be 2D array, got shape {data.shape}")
        
        if data.shape[1] != self.dim:
            raise ValueError(f"Data dimension {data.shape[1]} does not match index dimension {self.dim}")
        
        # Add to index
        self.index.add(data)
    
    def train(self, data):
        """
        Train the index (for trainable index types).
        For Flat indexes, this is equivalent to add().
        
        Args:
            data: numpy array of shape (n_vectors, dim) with dtype float32
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if len(data.shape) != 2:
            raise ValueError(f"Data must be 2D array, got shape {data.shape}")
        
        if data.shape[1] != self.dim:
            raise ValueError(f"Data dimension {data.shape[1]} does not match index dimension {self.dim}")
        
        # For flat indexes, training is the same as adding
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(data)
        else:
            self.index.add(data)
    
    def search(self, query, k=1):
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector(s) as numpy array of shape (n_queries, dim) or (dim,)
            k: Number of nearest neighbors to return (default: 1)
        
        Returns:
            List of tuples (distances, indices) for each query.
            - distances: numpy array of shape (k,) with distances to nearest neighbors
            - indices: numpy array of shape (k,) with indices of nearest neighbors
            If multiple queries, returns list of such tuples.
        """
        # Validate and prepare query
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        
        # Handle single query vector
        single_query = False
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
            single_query = True
        
        if len(query.shape) != 2:
            raise ValueError(f"Query must be 1D or 2D array, got shape {query.shape}")
        
        if query.shape[1] != self.dim:
            raise ValueError(f"Query dimension {query.shape[1]} does not match index dimension {self.dim}")
        
        if self.ntotal == 0:
            raise ValueError("Index is empty. Add vectors before searching.")
        
        if k > self.ntotal:
            k = self.ntotal
            print(f"Warning: k={k} is larger than number of vectors. Using k={self.ntotal}")
        
        # Perform search
        distances, indices = self.index.search(query, k)
        
        # Format results
        if single_query:
            return [(distances[0], indices[0])]
        else:
            return [(distances[i], indices[i]) for i in range(len(distances))]
    
    def save(self, filepath):
        """
        Save the index to disk.
        
        Args:
            filepath: Path to save the index file
        """
        # If using GPU, get CPU index for saving
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        else:
            index_to_save = self.index
        
        faiss.write_index(index_to_save, filepath)
    
    def load(self, filepath):
        """
        Load an index from disk.
        
        Args:
            filepath: Path to the index file
        """
        # Load index
        loaded_index = faiss.read_index(filepath)
        
        # Move to GPU if needed
        if self.use_gpu:
            if not faiss.get_num_gpus():
                raise RuntimeError("No GPU available. Set use_gpu=False or install GPU-enabled Faiss.")
            if self.gpu_res is None:
                self.gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, self.gpu_id, loaded_index)
        else:
            self.index = loaded_index
    
    @property
    def ntotal(self):
        """Get the total number of vectors in the index."""
        return self.index.ntotal
    
    @property
    def is_trained(self):
        """Check if the index is trained."""
        if hasattr(self.index, 'is_trained'):
            return self.index.is_trained
        return True  # Flat indexes are always "trained"
    
    def reset(self):
        """Reset the index, removing all vectors."""
        self.index.reset()
    
    def reconstruct(self, id):
        """
        Reconstruct a vector from its ID.
        
        Args:
            id: Vector ID in the index
        
        Returns:
            Reconstructed vector as numpy array
        """
        if not hasattr(self.index, 'reconstruct'):
            raise NotImplementedError("This index type does not support vector reconstruction")
        
        vector = self.index.reconstruct(int(id))
        return vector
    
    def reconstruct_batch(self, ids):
        """
        Reconstruct multiple vectors from their IDs.
        
        Args:
            ids: List or array of vector IDs
        
        Returns:
            Numpy array of reconstructed vectors
        """
        if not hasattr(self.index, 'reconstruct_batch'):
            # Fallback to individual reconstruction
            return np.array([self.reconstruct(id) for id in ids])
        
        ids = np.array(ids, dtype=np.int64)
        vectors = self.index.reconstruct_batch(ids)
        return vectors
