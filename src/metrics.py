import logging
import numpy as np
from rdkit import Chem
import torch


def canonicalize_smiles(smi):
    try:
        return Chem.CanonSmiles(smi)
    except:
        return ''

def predict_smiles(model, tokenizer, spc_img, num_beams=20, batch_size=100, \
                   do_sample=False, canonicalize=True):
    assert spc_img.ndim == 4, "Shape of input must be batches x channels x width x length"
    
    num_samples = len(spc_img)
    n_blocks = num_samples // batch_size + 1
    batched_spc_img = np.array_split(spc_img, n_blocks)
    
    smi_lst = []
    for batch in batched_spc_img:
        generated_ids = model.generate(
            inputs=batch,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=do_sample
        )
        
        batch_smi_lst = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        smi_lst.extend(batch_smi_lst)

    if canonicalize:
        return np.array(np.split(np.array([canonicalize_smiles(smi) \
                        for smi in smi_lst]), num_samples))
    else:
        return np.array(np.split(np.array(smi_lst), num_samples))


def calc_topk(ic_model, tokenizer, test_dataset, topks=[1, 5, 10, 15], 
                    num_beams=15, test_first=None, batch_size=100):
                        
    n_matched = 0
    unmatched_idx = []
    n_records = len(test_dataset)

    if test_first is not None:
        n_records = test_first

    # extract images for generation
    test_img = torch.cat([test_dataset[idx]['pixel_values'].unsqueeze(0) \
                  for idx in range(n_records)], dim=0)
    
    y_true = np.array([Chem.CanonSmiles(test_dataset[idx]['smiles']) \
                  for idx in range(n_records)])

    logging.info("Generating predictions with %s return sequences...", num_beams)

    y_pred = predict_smiles(ic_model, tokenizer, test_img, num_beams=num_beams,
                  batch_size=batch_size, do_sample=False, canonicalize=True)

    perf = dict()
    unmatched = {}

    for this_k in topks:
        logging.info("Calculating top-%d accuracy...", this_k)

        unmatched_idx = []
        n_matched = 0

        for idx in range(n_records):
            if y_true[idx] in y_pred[idx][:this_k]:
                n_matched += 1
            else:
                unmatched_idx.append(idx)

        acc = n_matched / n_records
        perf[this_k] = acc
        unmatched[this_k] = unmatched_idx

    return perf, unmatched, y_true, y_pred