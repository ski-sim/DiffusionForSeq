# Import public modules
import ml_collections
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Import custom modules
from . import utils

import torch.nn.functional as F
import torch.nn as nn
import copy

def count_params(model):
    return sum([p.numel() for p in model.parameters()])

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class DenoisingModel_CNN(torch.nn.Module):
    """
    CNN-based Denoising Model for discrete diffusion, adapted from the original CNNModel
    to fit the existing codebase structure.
    """
    
    def __init__(self, 
                 cfg: ml_collections.ConfigDict, 
                 time_encoder: object, 
                 logger: Optional[object] = None) -> None:
        """
        Args:
            cfg (ml_collections.ConfigDict): Config dictionary.
            time_encoder (object): Time encoder object.
            logger (None or object): Optional logger object.
                If None, no logger is used.
                (Default: None)
        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.D = cfg.data.shape  # Sequence length
        self.S = cfg.data.S      # Alphabet size (vocabulary size)
        self.pad_index = cfg.data.pad_index
        self.logger = logger
        
        # CNN specific configurations
        self.hidden_dim = cfg.denoising_model.get('hidden_dim', 256)
        self.num_cnn_stacks = cfg.denoising_model.get('num_cnn_stacks', 2)
        self.p_dropout = cfg.denoising_model.get('p_dropout', 0.1)
        self.eps = float(cfg.denoising_model.get('eps', 1e-8))
        self.stack_time = cfg.denoising_model.get('stack_time', True)
        
        # add cfg for classifier-free
        self.p_uncond = cfg.training.denoising_model.get('p_uncond', 0.1)
        self.condition_dim = cfg.denoising_model.get('condition_dim', 16)
        
        self.condition_encoder = nn.Sequential(
            nn.Linear(1, self.condition_dim),
            nn.ReLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.ReLU(),
            nn.Linear(self.condition_dim, self.hidden_dim)
        )
        
        self.time_cond_embedder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.uncond_embedding = nn.Parameter(torch.randn(self.hidden_dim))
        
        
        self.display(f"CNN Denoising Model - D: {self.D}, S: {self.S}")
        self.display(f"Stack time to x as CNN denoising model input: {self.stack_time}")

        # Set time encoder
        self.time_encoder = time_encoder
        self.encode_t = lambda t: time_encoder(t)
        
        # Input embedding layer
        # Convert one-hot encoded input to hidden dimension
        self.linear = nn.Conv1d(self.S, self.hidden_dim, kernel_size=9, padding=4)
        
        # Time embedding
        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Define CNN layers with different dilation rates
        self.num_layers = 5 * self.num_cnn_stacks
        base_convs = [
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=64, padding=256),
        ]
        
        # Replicate the base conv layers for num_cnn_stacks times
        self.convs = nn.ModuleList([
            copy.deepcopy(layer)
            for layer in base_convs
            for i in range(self.num_cnn_stacks)
        ])
        
        # Time conditioning layers
        self.time_layers = nn.ModuleList([
            Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Final output layers
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.S, kernel_size=1),  # Output to alphabet size
        )
        
        # Dropout
        self.dropout = nn.Dropout(self.p_dropout)

    def display(self, msg: str) -> None:
        """
        Display message either as logging info or as print if no logger has been defined.
        
        Args:
            msg (str): Message to be displayed.
        """
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    @property
    def device(self) -> object:
        """
        Return the device the model parameters are on.
        
        Returns:
            (object): The device the model parameters are on.
        """
        # Pick the first parameter and return on which device it is on
        return next(self.parameters()).device
    
    @property
    def num_parameters(self) -> int:
        """
        Return the number of model parameters.
        
        Return:
            (int): Number of model parameters.
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_x(self, x: torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)
        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()

    def forward(self, 
                xt: torch.tensor, 
                t: torch.tensor,
                condition: Optional[torch.tensor] = None) -> torch.tensor:
        """
        Define forward pass of the CNN denoising model.
        
        Args:
            xt (torch.tensor): Shape (B, D) - noised input sequences.
            t (torch.tensor): Shape (B,) - time steps.

        Return:
            (torch.tensor): Logits of shape (B, D, S).
        """
        # Extract the batch size
        B = xt.shape[0]
        device = xt.device
        
        # Encode the input sequence xt
        if condition is not None:
            if len(condition.shape) == 1:
                # If condition is a single value, expand it to match batch size
                condition = condition.unsqueeze(-1)
            
            # 휴 다행히 p_uncond는 여기 있었네
            if self.training:
                uncond_mask = torch.rand(B, device=device) < self.p_uncond
                cond_emb = self.condition_encoder(condition)
                cond_emb[uncond_mask] = self.uncond_embedding
            else:
                cond_emb = self.condition_encoder(condition)
        else:
            cond_emb = self.uncond_embedding.unsqueeze(0).repeat(B, 1)
        

        # One-hot encode the input
        if len(xt.shape) == 3:
            # Assume already one-hot encoded
            seq_encoded = xt  # (B, D, S)
        else:
            # Shape (B, D, S)
            seq_encoded = F.one_hot(xt.long(), num_classes=self.S).float()
        
        # Time embedding
        time_emb = F.relu(self.time_embedder(t))  # (B, hidden_dim)
        
        time_cond_emb = self.time_cond_embedder(
            torch.cat([time_emb, cond_emb], dim=-1)  # Concatenate time and condition embeddings
        )
        
        # Convert to conv1d format: (B, S, D)
        feat = seq_encoded.permute(0, 2, 1)  # (B, S, D)
        feat = F.relu(self.linear(feat))      # (B, hidden_dim, D)

        # Pass through CNN layers with time conditioning
        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            
            # Add time conditioning
            if self.stack_time:
                time_cond = self.time_layers[i](time_cond_emb)[:, :, None]  # (B, hidden_dim, 1)
                h = h + time_cond  # Broadcast along sequence dimension
            
            # Layer normalization (need to transpose for LayerNorm)
            h = self.norms[i](h.permute(0, 2, 1))  # (B, D, hidden_dim)
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))  # (B, hidden_dim, D)
            
            # Residual connection
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h

        # Final convolution to get logits
        feat = self.final_conv(feat)  # (B, S, D)
        
        # Convert back to (B, D, S) format to match expected output
        logits = feat.permute(0, 2, 1)  # (B, D, S)
        
        return logits


def activation_fn_factory(model_cfg: ml_collections.ConfigDict) -> object:
    """
    Factory that returns a torch activation function object
    based on the passed input config dictionary.

    Args:
        model_cfg (ml_collections.ConfigDict): Model specific config
            dictionary that should contain 'model_cfg.activation_fn'
            as entry, which is used to construct the torch activation
            function object.
    Return:
        (object): Torch activation function object.
    """
    # Use a ReLU activation function as default
    if 'activation_fn' in model_cfg:
        activation_fn_name = model_cfg.activation_fn.name
        if 'params' in model_cfg.activation_fn:
            activation_fn_params = model_cfg.activation_fn.params
        else:
            activation_fn_params = None
    else:
        # Use a ReLU activation function as default if 
        # the activation function is not defined in the 
        # model config
        activation_fn_name = 'ReLU'
        activation_fn_params = None

    # Try to get a handle on the activation function
    try:
        activation_fn_handle = getattr(torch.nn, activation_fn_name)
    except AttributeError:
        err_msg = f"There is no activation function in torch.nn with name '{activation_fn_name}'."
        raise ValueError(err_msg)
    
    # Initialize the activation function with parameters if specified
    if activation_fn_params is None:
        # Initialize without parameters
        return activation_fn_handle()
    else:
        # Initialize with parameters
        return activation_fn_handle(**activation_fn_params)

# Define the denoising model
class DenoisingModel(torch.nn.Module):
    def __init__(self, 
                 cfg:ml_collections.ConfigDict, 
                 time_encoder:object, 
                 logger:Optional[object]=None) -> None:
        """
        Args:
            cfg (ml_collections.ConfigDict): Config dictionary.
            time_encoder (object): Time encoder object.
            logger (None or object): Optional logger object.
                If None, no logger is used.
                (Default: None)

        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        # D,S,pad_index가 꼭 필요한데 왜 config 파일에서 null 처리되고 코드가 구동될까?
        self.D           = cfg.data.shape
        self.S           = cfg.data.S
        
        print(f"self.D: {self.D}, self.S: {self.S}")
        self.pad_index   = cfg.data.pad_index
        self.num_hidden  = cfg.denoising_model.num_hidden # Number of hidden layers
        self.hidden_dim  = cfg.denoising_model.hidden_dim # Dimension of each hidden layer
        self.p_dropout   = cfg.denoising_model.p_dropout
        self.eps         = float(cfg.denoising_model.eps)
        self.stack_time  = cfg.denoising_model.stack_time
        self.logger      = logger
        
        # Construct self.hidden_dims as list with self.num_hidden number elements
        # where each element corresponds to self.hidden_dim
        self.hidden_dims = [self.hidden_dim]*self.num_hidden

        self.display(f"Stack time to x as denoising model input: {self.stack_time}")

        # Set class attributes
        self.encode_t = lambda t: time_encoder(t)
        
        # (04/15) Define the time encoder
        self.time_encoder = time_encoder
        
        # Define input dimension
        self.input_dim = self.D*self.S
        if self.stack_time:
            # If the time should be stacked, add the time 
            # encoding dimension to the input dimension
            self.input_dim += self.time_encoder.dim
        
        # Define the output dimension
        self.output_dim = self.D*self.S 

        # Define the linear parts of the model
        linear_list       = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        linear_list      += [torch.nn.Linear(self.hidden_dims[layer_id-1], self.hidden_dims[layer_id]) for layer_id in range(1, len(self.hidden_dims))]
        self.linear_list  = torch.nn.ModuleList(linear_list)
        self.linear_last  = torch.nn.Linear(self.hidden_dims[-1], self.output_dim)

        # Define an activation function
        self.activation_fn = activation_fn_factory(cfg.denoising_model)

        # Define (global) dropout function
        self.dropout_fn = torch.nn.Dropout(p=self.p_dropout, inplace=False)


    def display(self, msg:str) -> None:
        """
        Display message either as logging info or as print if no logger has been defined. 
        
        Args:
            msg (str): Message to be displayed.
        
        """
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    @property
    def device(self) -> object:
        """
        Return the device the model parameters are on.
        
        Returns:
            (object): The device the model parameters are on.
        
        """
        # Pick the first parameter and return on which device it is on
        return next(self.parameters()).device
    
    @property
    def num_parameters(self) -> int:
        """
        Return the number of model parameters.
        
        Source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/10

        Return:
            (int): Number of model parameters.

        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_x(self, x:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)

        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()

    def forward(self, 
                xt:torch.tensor, 
                t:torch.tensor) -> torch.tensor:
        """"
        Define forward pass of the model.
        
        Args:
            xt (torch.tensor): Shape (B, D).
            t (torch.tensor): Shape (B,).

        Return:
            (torch.tensor): Logits of shape (B, D, S).
        
        """
        # Extract the batch size
        B = xt.shape[0]

        # Encode space and flatten from (B, D, S) to (B, D*S)
        xt_enc = self.encode_x(xt) # (B, D, S)
        xt_enc = xt_enc.view(-1, self.D*self.S) # (B, D*S)

        # Stack encoded time to h if requested
        if self.stack_time:
            t_enc = self.encode_t(t)
            h = torch.cat([xt_enc, t_enc], dim=-1) 
        else:
            h = xt_enc

        # Perform pass through the network
        for layer_id in range(len(self.linear_list)):
            h = self.dropout_fn(h)
            h = self.linear_list[layer_id](h)
            h = self.activation_fn(h)

        # Shape (B, #classes)
        h = self.dropout_fn(h)
        h = self.linear_last(h)
    
        # Bring logits in correct shape
        logits = h.view(-1, self.D, self.S) # (B, D, S)
    
        return logits


class DiscretePredictorGuideModel(torch.nn.Module):
    output_type = 'class_distribution'
    
    def __init__(self, 
                 num_classes:int, 
                 output_layer_dim:int, 
                 model_cfg:ml_collections.ConfigDict, 
                 general_cfg:ml_collections.ConfigDict, 
                 time_encoder:object, 
                 logger:Optional[object]=None) -> None:
        """
        Args:
            num_classes (int): Number of classes.
            output_layer_dim (int): Dimension of the output layer.
            model_cfg (ml_collections.ConfigDict): Model specific config dictionary.
            general_cfg (ml_collections.ConfigDict): General (non-model specific) config dictionary.
            time_encoder (object): Time encoder object.
            logger (None or object): Optional logger object.
                If None, no logger is used.
                (Default: None)

        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.D                = general_cfg.data.shape
        self.S                = general_cfg.data.S
        self.y_guide_name     = model_cfg.y_guide_name
        self.hidden_dims      = model_cfg.hidden_dims
        self.p_dropout        = model_cfg.p_dropout
        self.eps              = float(model_cfg.eps)
        self.stack_time       = model_cfg.stack_time
        self.y_enc_dim        = num_classes
        self.time_encoder     = time_encoder
        self.output_layer_dim = output_layer_dim
        self.logger           = logger

        self.display(f"Stack time to x as {self.y_guide_name}-predictor model input: {self.stack_time}")

        # Define input dimension
        self.input_dim = self.D*self.S
        if self.stack_time:
            # If the time should be stacked, add the time 
            # encoding dimension to the input dimension
            self.input_dim += self.time_encoder.dim

        # Define the t encoder
        self.encode_t = lambda t: self.time_encoder(t)

        # Define the model parts
        linear_list       = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        linear_list      += [torch.nn.Linear(self.hidden_dims[layer_id-1], self.hidden_dims[layer_id]) for layer_id in range(1, len(self.hidden_dims))]
        self.linear_list  = torch.nn.ModuleList(linear_list)
        self.linear_last  = torch.nn.Linear(self.hidden_dims[-1], self.output_layer_dim) # Output layer

        # Define an activation function
        self.activation_fn = activation_fn_factory(model_cfg)

        # Define (global) dropout function
        self.dropout_fn = torch.nn.Dropout(p=self.p_dropout, inplace=False)
    
    @property
    def device(self) -> object:
        """
        Return the device the model parameters are on.
        
        Returns:
            (object): The device the model parameters are on.
        
        """
        # Pick the first parameter and return on which device it is on
        return next(self.parameters()).device
    
    @property
    def num_parameters(self) -> int:
        """
        Return the number of model parameters.
        
        Source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/10

        Return:
            (int): Number of model parameters
            
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


    @property
    def num_categories(self) -> int:
        """
        Return the number of categories that correspond to the dimensionality
        of the one-hot encoded y.

        Return:
            (int): Number of categories.

        """
        return self.y_enc_dim

    def display(self, msg:str) -> None:
        """
        Display message either as logging info or as print if no logger has been defined. 
        
        Args:
            msg (str): Message to be displayed.
        
        """
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    def encode_x(self, x:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)

        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()
    
    def encode_y(self, y:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor y.

        Args:
            y (torch.tensor): Torch tensor of shape (B,) holding
                categorical property entries, where B is the batch 
                size.

        Return:
            (torch.tensor): Tensor where the categorical entries 
                (of which there are self.num_categoricals different one) 
                of y have been one-hot encoded to a tensor of shape 
                (B, self.num_categoricals).

        """
        return torch.nn.functional.one_hot(y.long(), num_classes=self.num_categories).float()

    def forward(self, 
                batch_data_t:dict, 
                t:torch.tensor, 
                is_x_onehot:bool=False) -> torch.tensor:
        """
        Define forward pass of the model.
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): Probabilities for each component of the encoded property 'y'
                as 2D torch tensor of shape (batch_size, dim[encoded(y)])
        
        """
        # Get xt and t, encode both, and stack them
        xt = batch_data_t['x']
        B  = xt.shape[0]

        # Differ the cases where x is already encoded or not
        if is_x_onehot:
            # xt is already encoded
            xt_enc = xt # (B, D, S)
        else:
            # xt has to be encoded
            xt_enc = self.encode_x(xt) # (B, D, S)

        # Flatten features
        xt_enc = xt_enc.view(-1, self.D*self.S) # (B, D*S)

        # Stack encoded time to h if requested
        if self.stack_time:
            t_enc = self.encode_t(t).view(-1, 1)
            h = torch.cat([xt_enc, t_enc], dim=-1) 
        else:
            h = xt_enc

        # Perform pass through the network
        for layer_id in range(len(self.linear_list)):
            h = self.dropout_fn(h)
            h = self.linear_list[layer_id](h)
            h = self.activation_fn(h)
        
        # Perform pass through last layer
        h = self.dropout_fn(h)
        h = self.linear_last(h) # (B, output_layer_dim)

        # Determine the class probabilities from the output layer
        # 왜 마지막에 simplex로 안만들어주지?
        p_y = self._get_p_y_from_output_layer(h, t)
        return p_y
    
    def _get_p_y_from_output_layer(self, 
                                   h:torch.tensor, 
                                   t:torch.tensor) -> torch.tensor:
        """
        Map the values from the output layer (i.e. self.last_linear) to
        class propabilities.

        Args:
            h (torch.tensor): Output layer values.
            t (torch.tensor): Times.
        
        Return:
            (torch.tensor): Class probabilities p(y|h(x),t).
        
        """
        raise NotImplementedError("The method '_get_p_y_from_output_layer' has not been implemented.")

    def log_prob(self, 
                 batch_data_t:torch.tensor, 
                 t:torch.tensor, 
                 is_x_onehot:bool=False) -> torch.tensor:
        """
        Return the log probability given the data. 
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): (Batched) log-probability for each point in the batch as
                1D torch tensor of shape (batch_size,).
        
        """
        y_data = batch_data_t[self.y_guide_name]
        xt = batch_data_t['x']

        # Encode the property input y (class index)
        # Shape (B, #classes)
        y_data_enc = self.encode_y(y_data).to(xt.device)

        # Determine the class-probabilities
        p_y_pred = self.forward(batch_data_t, t, is_x_onehot=is_x_onehot) # Shape (B, #classes)

        # Calculate the categorical log-probability for each point and each
        # category/feature and then sum over the feature (i.e. the second) axis
        log_prob = torch.sum(y_data_enc*torch.log(p_y_pred+self.eps), dim=-1)

        return log_prob
    
class CategoricalPredictorGuideModel(DiscretePredictorGuideModel):
    def __init__(self, 
                 num_categories:int, 
                 **kwargs) -> None:
        """
        Args:
            num_categories (int): Number of mutually exclusive categories.
            **kwargs: Keyword arguments forwarded to parent class.

        """

        # Initialize the parent class
        # Remark: For categorical, the dimension of the output layer 
        #         corresponds to the number of categories.
        super().__init__(num_classes=num_categories, output_layer_dim=num_categories, **kwargs)

        # Define the softmax function that should be applied along the
        # second axis (i.e. the feature axis)
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    def _get_p_y_from_output_layer(self, 
                                   h:torch.tensor, 
                                   t:torch.tensor) -> torch.tensor:
        """
        Map the values from the output layer (i.e. self.last_linear) to
        class propabilities.

        Remark:
        Here, use a softmax to transform output layer values h 
        (in R) to class probabilities of shape (B, #categories).

        Args:
            h (torch.tensor): Output layer values.
            t (torch.tensor): Times.
        
        Return:
            (torch.tensor): Class probabilities p(y|h(x),t).
        
        """
        return self.softmax_fn(h)
    

class OrdinalPredictorGuideModel(DiscretePredictorGuideModel):    
    def __init__(self, 
                 num_ordinals:int, 
                 sigma_noised:float=1.0,
                 **kwargs) -> None:
        """
        Args:
            num_ordinals (int): Number of ordinals.
            sigma_noised (float): Sigma of the properties 
                (for which a predictor is setup here) of 
                the fully noised samples (at t=0).
                (Default: 1.0)
            **kwargs: Keyword arguments forwarded to parent class.

        """
        # Initialize the parent class
        # Remark: For ordinal, the dimension of the output layer corresponds
        #         to 1 (mean of discretized normal distribution).
        super().__init__(num_classes=num_ordinals, output_layer_dim=1, **kwargs)

        # Define ordinal model specific attributes
        self.sigma_noised       = sigma_noised
        self.log_sigma_unnoised = torch.nn.Parameter(torch.log(torch.tensor(sigma_noised))) # Initialize equal to log(sigma_noised)

    @property
    def sigma_unnoised(self) -> torch.tensor:
        """
        Return the unnoised sigma based on the model parameter 'self.log_sigma_unnoised'.

        Return:
            (torch.tensor): Unnoised sigma.

        """
        return torch.exp(self.log_sigma_unnoised)

    def get_sigma_t(self, t:torch.tensor) -> torch.tensor:
        """
        Return sigma(t) as a linear interpolation between the
        noised sigma at t=0 and the unnoised sigma at t=1.
        
        Remark: 
        At t=1 we have the data distribution (unnoised)
        and at t=0 we have the noised distribution.
        Thus, interpolate the predictor sigma in a similar way.

        Args:
            t (torch.tensor): Times.

        Return:
            (torch.tensor): Determined sigma(t).
        
        """
        return t*self.sigma_unnoised+(1-t)*self.sigma_noised
    
    def plot_sigma_t(self) -> object:
        """
        Plot sigma(t) vs. t and return the resulting figure.
        
        Return:
            (object): Matplotlib figure object.
        
        """
        fig = plt.figure()
        t = torch.linspace(0, 1, 1000).to(self.sigma_unnoised.device)
        sigma_t = self.get_sigma_t(t)
        plt.plot(utils.to_numpy(t), utils.to_numpy(sigma_t), 'b-')
        plt.xlabel('t')
        plt.ylabel('sigma(t)')
        plt.xlim([0, 1])
        plt.ylim([0, max(utils.to_numpy(sigma_t))*1.05])
        plt.show()
        return fig

    def _get_p_y_from_output_layer(self, 
                                   mu:torch.tensor,
                                   t:torch.tensor) -> torch.tensor:
        """
        Map the values from the output layer (i.e. self.last_linear) to
        class propabilities.

        Remark:
        Here, the output layer values correspond to the mean 'mu' of a discretized 
        normal distribution that is used as ordinal distribution.
        Thus, calculate the class probabilities by integrating the normal distribution
        within the ordinal bounaries.
        The standard deviation (sigma) is determined from the passed time 't' (see below).

        Args:
            mu (torch.tensor): Output layer values (that determines to the mean of a 
                discretized normal distribution) of shape (B, 1).
            t (torch.tensor): Times.
        
        Return:
            (torch.tensor): Class probabilities p(y|mu(x),t).
        
        """
        # Define the boundaries of the ordinals
        # Remark: The ordinal indices are zero based ([0, 1, ..., num_ordinals-1]) so
        #         that the ordinal boundaries are [-0.5, 0.5, ..., (num_ordinals-1)+-0.5]
        #         which is equivalent to [-0.5, 0.5, ..., num_ordinals-0.5]
        num_ordinals = self.y_enc_dim
        ordinal_bounds = torch.linspace(-0.5, num_ordinals-0.5, num_ordinals+1).to(mu.device) # (#ordinals+1, )

        # Determine sigma(t)
        sigma_t = (self.get_sigma_t(t)+self.eps).reshape(-1, 1) # (B, 1)
       
        # Determine the intergrals of the normal distribution defined by h up to 
        # each of the ordinal boundaries
        cdfs = torch.distributions.normal.Normal(loc=mu, scale=sigma_t).cdf(ordinal_bounds) # (B, num_ordinals+1)
       
        # Determine the probability of each ordinal as the integral between each
        # of its boundaries
        ordinal_ints = cdfs[:, 1:]-cdfs[:, :-1] # (B, num_ordinals)
        
        # These integrals do not sum up to 1, because the contribution of the integrals over the normal 
        # distribution for -inf to -0.5 (first lower-bound) and from num_ordinals-0.5 (last upper-bound) is 
        # not included (and because only the integral from -inf to inf gives 1).
        # Thus, normalize these integrals to obtain the ordinal probabilities
        ordinal_probs = ordinal_ints/(torch.sum(ordinal_ints, dim=-1).reshape(-1, 1)+self.eps) # (B, num_ordinals)

        return ordinal_probs


class NormalPredictorGuideModel(torch.nn.Module):
    output_type = 'continuous_value'

    def __init__(self, 
                 model_cfg:ml_collections.ConfigDict, 
                 general_cfg:ml_collections.ConfigDict, 
                 time_encoder:object, 
                 sigma_noised:float=1.0, 
                 logger:Optional[object]=None) -> None:
        """
        Args:
            model_cfg (ml_collections.ConfigDict): Model specific config dictionary.
            general_cfg (ml_collections.ConfigDict): General (non-model specific) config dictionary.
            time_encoder (object): Time encoder object.
            sigma_noised (float): Sigma of the properties 
                (for which a predictor is setup here) 
                of the fully noised samples (at t=0).
                (Default: 1.0)
            logger (None or object): Optional logger object.
                If None, no logger is used.
                (Default: None)

        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.D                  = general_cfg.data.shape
        self.S                  = general_cfg.data.S
        self.y_guide_name       = model_cfg.y_guide_name
        self.hidden_dims        = model_cfg.hidden_dims
        self.stack_time         = model_cfg.stack_time
        self.p_dropout          = model_cfg.p_dropout
        self.eps                = float(model_cfg.eps)
        self.time_encoder       = time_encoder
        self.logger             = logger
        self.output_layer_dim   = 1
        # proposal distribution은 시그마 값이 1
        self.sigma_noised       = sigma_noised
        self.log_sigma_unnoised = torch.nn.Parameter(torch.log(torch.tensor(sigma_noised))) # Initialize equal to log(sigma_noised)

        self.display(f"Stack time to x as {self.y_guide_name}-predictor model input: {self.stack_time}")

        # Define input dimension
        self.input_dim = self.D*self.S
        if self.stack_time:
            # If the time should be stacked, add the time 
            # encoding dimension to the input dimension
            self.input_dim += self.time_encoder.dim

        # Define the t encoder
        self.encode_t = lambda t: self.time_encoder(t)

        # Define the model parts
        linear_list       = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        linear_list      += [torch.nn.Linear(self.hidden_dims[layer_id-1], self.hidden_dims[layer_id]) for layer_id in range(1, len(self.hidden_dims))]
        self.linear_list  = torch.nn.ModuleList(linear_list)
        self.linear_last  = torch.nn.Linear(self.hidden_dims[-1], self.output_layer_dim) # Output layer

        # Define an activation function
        self.activation_fn = activation_fn_factory(model_cfg)

        # Define (global) dropout function
        self.dropout_fn = torch.nn.Dropout(p=self.p_dropout, inplace=False)
    
    @property
    def device(self) -> object:
        """
        Return the device the model parameters are on.
        
        Returns:
            (object): The device the model parameters are on.
        
        """
        # Pick the first parameter and return on which device it is on
        return next(self.parameters()).device
    
    @property
    def num_parameters(self) -> int:
        """
        Return the number of model parameters.
        
        Source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/10

        Return:
            (int): Number of model parameters.
        
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_x(self, x:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)

        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()
    
    def display(self, msg:str) -> None:
        """
        Display message either as logging info or as print if no logger has been defined. 
        
        Args:
            msg (str): Message to be displayed.
        
        """
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
            
    def forward(self, 
                batch_data_t:torch.tensor, 
                t:torch.tensor, 
                is_x_onehot:bool=False) -> torch.tensor:
        """
        Define forward pass of the model.
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): Probabilities for each component of the encoded property 'y'
                as 2D torch tensor of shape (batch_size, dim[encoded(y)])
        
        """
        # Get xt and t, encode both, and stack them
        xt = batch_data_t['x']
        B = xt.shape[0]

        # Differ the cases where x is already encoded or not
        if is_x_onehot:
            # xt is already encoded
            xt_enc = xt # (B, D, S)
        else:
            # xt has to be encoded
            xt_enc = self.encode_x(xt) # (B, D, S)

        # Flatten features
        xt_enc = xt_enc.view(-1, self.D*self.S) # (B, D*S)

        # Stack encoded time to h if requested
        if self.stack_time:
            t_enc = self.encode_t(t).view(-1, 1)
            h = torch.cat([xt_enc, t_enc], dim=-1) 
        else:
            h = xt_enc

        # Perform pass through the network
        for layer_id in range(len(self.linear_list)):
            h = self.dropout_fn(h)
            h = self.linear_list[layer_id](h)
            h = self.activation_fn(h)
        
        # Perform pass through last layer
        h = self.dropout_fn(h)
        h = self.linear_last(h) # (B, #classes)

        return h.squeeze()
    
    @property
    def sigma_unnoised(self) -> torch.tensor:
        """
        Return the unnoised sigma based on the model parameter 'self.log_sigma_unnoised'.

        Return:
            (torch.tensor): Unnoised sigma.

        """
        return torch.exp(self.log_sigma_unnoised)

    def get_sigma_t(self, t:torch.tensor) -> torch.tensor:
        """
        Return sigma(t) as a linear interpolation between the
        noised sigma at t=0 and the unnoised sigma at t=1.
        
        Remark: 
        At t=1 we have the data distribution (unnoised)
        and at t=0 we have the noised distribution.
        Thus, interpolate the predictor sigma in a similar way.

        Args:
            t (torch.tensor): Times.

        Return:
            (torch.tensor): Determined sigma(t).
        
        """
        return t*self.sigma_unnoised+(1-t)*self.sigma_noised

    def plot_sigma_t(self) -> object:
        """
        Plot sigma(t) vs. t and return the resulting figure.
        
        Return:
            (object): Matplotlib figure object.
        
        """
        fig = plt.figure()
        t = torch.linspace(0, 1, 1000).to(self.sigma_unnoised.device)
        sigma_t = self.get_sigma_t(t)
        plt.plot(utils.to_numpy(t), utils.to_numpy(sigma_t), 'b-')
        plt.xlabel('t')
        plt.ylabel('sigma(t)')
        plt.xlim([0, 1])
        # plt.ylim([0, max(utils.to_numpy(sigma_t))*1.05])
        plt.show()
        return fig

    
    def log_prob(self, 
                 batch_data_t:torch.tensor, 
                 t:torch.tensor, 
                 is_x_onehot:bool=False) -> torch.tensor:
        """
        Return the log probability given the data. 
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): (Batched) log-probability for each point in the batch as
                1D torch tensor of shape (batch_size,).
        
        """
        y_data = batch_data_t[self.y_guide_name]
 
        # Determine the class-probabilities
        y_pred = self.forward(batch_data_t, t, is_x_onehot=is_x_onehot) # Shape (B, #classes)
       
        # print(t)
        # print(self.get_sigma_t(t))
        # print('#'*50)
        # assert False
        # Determine sigma(t) and log(sigma(t))
        sigma_t     = self.get_sigma_t(t).squeeze()+self.eps
        log_sigma_t = torch.log(sigma_t)

        # Calculate the log_prob per point
        square_diff = (y_data.squeeze()-y_pred.squeeze())**2/(2*sigma_t**2)
        log_prob = -square_diff-log_sigma_t-np.sqrt(2*np.pi)


        return log_prob
    

def activation_fn_factory(model_cfg:ml_collections.ConfigDict) -> object:
    """
    Factory that returns a torch activation function object
    based on the passed input config dictionary.

    Args:
        model_cfg (ml_collections.ConfigDict): Model specific config
            dictionary that should contain 'model_cfg.activation_fn'
            as entry, which is used to construct the torch activation
            function object.
    Return:
        (object): Torch activation function object.

    """
    # Use a ReLU activation function as default
    if 'activation_fn' in model_cfg:
        activation_fn_name = model_cfg.activation_fn.name
        if 'params' in model_cfg.activation_fn:
            activation_fn_params = model_cfg.activation_fn.params
        else:
            activation_fn_params = None
    else:
        # Use a ReLU activation function as default if 
        # the activation function is not defined in the 
        # model config
        activation_fn_name   = 'ReLU'
        activation_fn_params = None

    # Try to get a handle on the activation function
    try:
        activation_fn_handle = getattr(torch.nn, activation_fn_name)
    except AttributeError:
        err_msg = f"There is no activation function in torch.nn with name '{activation_fn_name}'."
        raise ValueError(err_msg)
    
    # Initialize the activation function with parameters if specified
    if activation_fn_params is None:
        # Initialize without parameters
        return activation_fn_handle()
    else:
        # Initialize with parameters
        return activation_fn_handle(**activation_fn_params)