from detanet_model import (nn_vib_analysis, nmr_calculator, 
                           Lorenz_broadening, uv_model)
from ir_smarts import EXTSMARTS
import os
import logging
from rdkit import Chem
from rdkit.Chem import rdqueries
from scipy.interpolate import interp1d
from tqdm import tqdm
import torch
import tensorflow as tf
from torch.utils.data import Dataset


class TransformPipeline:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, spc):
        for transform in self.transforms:
            spc = transform(spc)
        return spc    

    
class ToTransmittance:
    """
    Convert to Transimittance.
    Assuming l = 1 cm-1, w = 5 cm^-1. 
    Refer to https://www.researchgate.net/publication/279441865_Infrared_Intensity_and_Lorentz_Epsilon_Curve_from_'Gaussian'_FREQ_Output.
    T = 10 ** (-27.648 * I_IR / 5) = 10 ** (-5.5296 * I_IR)
    """
    
    def __init__(self, concentration=0.1):
        self._concentration = concentration
        
    def __call__(self, spc):
        return 10 ** (-5.5296 * self._concentration * spc)
    
    
class MaxNormalization:
    def __call__(self, spc):
        min_val = spc.min()
        return (spc - min_val) / (spc.max() - min_val)
    
class PadData:
    def __init__(self, target_length, pad_val):
        self.target_length = target_length
        self.pad_val = pad_val

    def __call__(self, spc):
        current_length = spc.shape[0]
        if current_length < self.target_length:
            padding = torch.zeros(self.target_length - current_length) + self.pad_val
            spc = torch.cat([spc, padding])
        return spc
    
class SpcToImage:
    def __init__(self, reshaped_size, desired_size):
        self.reshaped_size = reshaped_size
        self.desired_size = desired_size
    
    def __call__(self, spc):
        image = spc.view(self.reshaped_size)  # Reshape to 2D tensor
        image = image.unsqueeze(0).unsqueeze(0)  # Add a batch dimension

        # Scale image to a fixed size
        if self.reshaped_size != self.desired_size:
            image = torch.nn.functional.interpolate(image, size=self.desired_size, mode='nearest')
            
        image = image.squeeze(0)
        return image
    
    
def get_fg_encoding(smile, smarts):
    # Create queries from SMARTS patterns
    queries = [Chem.MolFromSmarts(val) for _, val in smarts.items()]
    mol = Chem.MolFromSmiles(smile)
    
    # Check substructure matches for each SMARTS pattern
    match_array = [mol.GetSubstructMatches(query) for query in queries]
    encoded_array = [1 if len(matches) > 0 else 0 for matches in match_array]
    
    return encoded_array


def gen_ir(z, pos, vib_model, x_axis, sigma):
    '''Calculation of frequency, IR intensity, Raman activity for each vibration mode. 
    (Units are consistent with Gaussian G16)'''
    freq, iir, _ = vib_model(z=z, pos=pos)

    '''Broadening of frequency and intensity'''
    yir = Lorenz_broadening(freq, iir, c=x_axis, sigma=sigma).detach().numpy()

    return yir


def gen_uv(z, pos, uv_model, x_axis):
    '''calculate uv spectrum'''
    model_xval = 1239.85/torch.linspace(1.5,13.5,240).detach().numpy()
    model_yval = uv_model(z=z, pos=pos).detach().numpy()
    yuv = interp1d(model_xval, model_yval)(x_axis)

    return yuv

def gen_nmr(z, pos, nmr_model, x_axis, sigma=0.05):
    '''calculate 1H NMR spectrum'''
    _, sh = nmr_model(pos=pos,z=z)
    inth = torch.ones_like(sh)
    ynmr = Lorenz_broadening(sh, inth, c=x_axis, sigma=sigma).detach().numpy()

    return ynmr


class IrDataset(Dataset):
    def __init__(self, data_list, 
                 ir_sigma=5, xir=torch.linspace(500, 4000, 3501), ir_cons = (1e-2,), 
                 xuv=torch.linspace(100, 400, 240), nmr_sigma=0.05, \
                 xnmr=torch.linspace(0, 12, 516), device=torch.device('cpu'), \
                 data_path=None, use_transmittance=False, ir_only=False, \
                 canonicalize=True, smarts=None, further_remove=[]):
        
        self.data_list = data_list
        self.ir_features = []
        self.ir_labels = []
        self.uv_features = []
        self.uv_labels = []
        self.nmr_features = []
        self.nmr_labels = []
        self.further_remove = further_remove
        self.device = device
        self.xir = xir
        self.xuv = xuv
        self.xnmr = xnmr
        self.ir_sigma = ir_sigma
        self.nmr_sigma = nmr_sigma
        self.data_path = data_path
        self.ir_cons = ir_cons
        
        self.common_labels = None
        self.common_ir = None
        self.common_uv = None
        self.common_nmr = None
        self.common_smiles = None
        self.spc = None
        
        self.ir_only = ir_only
        self.canonicalize = canonicalize
        self.smarts = smarts
        
        if use_transmittance:
            self.ir_pipelines = TransformPipeline([PadData(3600, 0.), ToTransmittance(0.1)])
        else:
            self.ir_pipelines = TransformPipeline([PadData(3600, 0.), MaxNormalization()])
            
        self.uv_pipelines = TransformPipeline([MaxNormalization()])
        self.nmr_pipelines = TransformPipeline([MaxNormalization()])
        
        if ir_only:
            self.final_pipelines = TransformPipeline([SpcToImage((60, 60), (60, 60))])
        else:
            self.final_pipelines = TransformPipeline([SpcToImage((66, 66), (66, 66))])

            
    def cal_ir(self):
        model = nn_vib_analysis(device=self.device, Linear=False, scale=0.965)
        
        for data in tqdm(self.data_list):                
            spc = torch.tensor(gen_ir(data.z, data.pos, model, self.xir, self.ir_sigma))                
            fg_encode = torch.tensor(get_fg_encoding(data.smile, EXTSMARTS))
            self.ir_features.append((data.smile, spc))
            self.ir_labels.append(fg_encode)
            
    def cal_uv(self):
        model = uv_model(device=self.device)
        
        for data in tqdm(self.data_list):                
            spc = torch.tensor(gen_uv(data.z, data.pos, model, self.xuv))
            fg_encode = torch.tensor(get_fg_encoding(data.smile, EXTSMARTS))
            self.uv_features.append((data.smile, spc))
            self.uv_labels.append(fg_encode)
            
            
    def cal_nmr(self):
        model = nmr_calculator(device=self.device)
        
        for data in tqdm(self.data_list):                
            spc = torch.tensor(gen_nmr(data.z, data.pos, model, self.xnmr, self.nmr_sigma)) 
            fg_encode = torch.tensor(get_fg_encoding(data.smile, EXTSMARTS))
            self.nmr_features.append((data.smile, spc))
            self.nmr_labels.append(fg_encode)
            
            
    def load(self):
        data_type = {
            "ir": [self.ir_features, self.ir_labels],
            "uv": [self.uv_features, self.uv_labels], 
            "nmr": [self.nmr_features, self.nmr_labels]
        }

        for this_type, _ in data_type.items():
            # Load features
            feature_file = os.path.join(self.data_path, f"{this_type}_features.pth")
            label_file = os.path.join(self.data_path, f"{this_type}_labels.pth")

            if os.path.isfile(feature_file) and os.path.isfile(label_file):
                logging.info(f"Loading {this_type} feature data from %s...", feature_file)
                features = torch.load(feature_file)
                setattr(self, f"{this_type}_features", features)

                # Load IR labels
                logging.info(f"Loading {this_type} label data from %s...", label_file)
                labels = torch.load(label_file)
                setattr(self, f"{this_type}_labels", labels)
                
        logging.info("Consolidating data...")
        self.consolidate_data()
        
        
    def save(self):
        data_type = {
            "ir": [self.ir_features, self.ir_labels],
            "uv": [self.uv_features, self.uv_labels], 
            "nmr": [self.nmr_features, self.nmr_labels]
        }
        
        for this_type, this_data in data_type.items():
            features, labels = this_data
            if features:
                feature_file = os.path.join(self.data_path, f"{this_type}_features.pth")
                logging.info(f"Saving {this_type} feature data to %s...", feature_file)
                torch.save(features, feature_file)

                label_file = os.path.join(self.data_path, f"{this_type}_labels.pth")
                logging.info(f"Saving {this_type} label data to %s...", label_file)           
                torch.save(labels, label_file)                                   
    
    @staticmethod
    def apply_pipelines(data, pipelines):
        if isinstance(data, list):
            return list(map(pipelines, data))
        else:
            return(pipelines(data))
    
    def consolidate_data(self):
        self.common_ir = [item[1] for item in self.ir_features]
        self.common_uv = [item[1] for item in self.uv_features]
        self.common_nmr = [item[1] for item in self.nmr_features]
        self.common_smiles = [item[0] for item in self.ir_features]
        self.common_labels = self.ir_labels
        
        bad_indices = []
        for idx in range(len(self.common_smiles)):
            if torch.isnan(self.common_ir[idx]).any() or torch.isnan(self.common_uv[idx]).any() \
                            or torch.isnan(self.common_nmr[idx]).any():
                bad_indices.append(idx)
                
        if bad_indices:
            logging.info("Removing %i bad records from data!!!", len(bad_indices))
            self.common_ir = self.remove_data(self.common_ir, bad_indices)
            self.common_uv = self.remove_data(self.common_uv, bad_indices)
            self.common_nmr = self.remove_data(self.common_nmr, bad_indices)
            self.common_smiles = self.remove_data(self.common_smiles, bad_indices)
            self.common_labels = self.remove_data(self.common_labels, bad_indices)
                    
        self.common_ir = self.apply_pipelines(self.common_ir, self.ir_pipelines)
        self.common_uv = self.apply_pipelines(self.common_uv, self.uv_pipelines)
        self.common_nmr = self.apply_pipelines(self.common_nmr, self.nmr_pipelines)
        
        # combine data
        if self.ir_only:
            spc = self.common_ir
        else:
            spc = []
            for idx in range(len(self.common_ir)):
                spc.append(torch.concat((self.common_ir[idx], self.common_uv[idx], self.common_nmr[idx])))
            
        self.spc = self.apply_pipelines(spc, self.final_pipelines)
        
        bad_indices = []
        for idx in range(len(self.spc)):
            if torch.isnan(self.spc[idx]).any():
                bad_indices.append(idx)
                
        if not bad_indices:
            bad_indices = self.further_remove
                
        if bad_indices:
            logging.info("Further removing %i bad records from data: %s", 
                                len(bad_indices), bad_indices)
            self.spc = self.remove_data(self.spc, bad_indices)
            self.common_ir = self.remove_data(self.common_ir, bad_indices)
            self.common_uv = self.remove_data(self.common_uv, bad_indices)
            self.common_nmr = self.remove_data(self.common_nmr, bad_indices)
            self.common_smiles = self.remove_data(self.common_smiles, bad_indices)
            self.common_labels = self.remove_data(self.common_labels, bad_indices)
            
        if self.smarts:
            for idx in range(len(self.common_smiles)):
                self.common_labels[idx] = torch.tensor(\
                            get_fg_encoding(self.common_smiles[idx], self.smarts))
            
    @staticmethod
    def remove_data(data_lst, bad_indices):
        return [item for idx, item in enumerate(data_lst) if idx not in bad_indices]
        
    def __len__(self):
        if not self.common_labels:
            self.consolidate_data()
            
        return len(self.common_labels)

    def __getitem__(self, idx):
        if not self.common_labels:
            self.consolidate_data()

        this_smiles = self.common_smiles[idx]
        if self.canonicalize:
            this_smiles = Chem.CanonSmiles(this_smiles)

        return self.common_labels[idx], this_smiles, self.spc[idx], self.common_ir[idx], self.common_uv[idx], self.common_nmr[idx]


def generate_ic_dataset(original_dataset, tokenizer, max_length=30):
    new_dataset = []
    for item in original_dataset:
        pixel_values = item[2]
        text = item[1]
        tokenizer_ret = tokenizer(text, truncation=True,
                    padding="max_length", max_length=max_length)
        labels = tokenizer_ret['input_ids']  # Tokenize the text
        attention_mask = tokenizer_ret['attention_mask']  # Generate attention mask

        new_item = {
            'pixel_values': torch.tensor(pixel_values),
            'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(attention_mask),
            'smiles': text
        }
        new_dataset.append(new_item)
    return new_dataset
   