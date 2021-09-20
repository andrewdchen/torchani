"""Adapted from energy_force.py"""

import torch
import torchani
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def read_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument
    add_arg('--data_path', required = True)

    return parser.parse_args()

def hartree_energy(mol, model, device):
    if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) == -1:  # optional random seed for reproducibility)
            print("Using 2D coordinates instead!")
            AllChem.Compute2DCoords(mol)
        
    #AllChem.MMFFOptimizeMolecule(mol)
    coordinates = mol.GetConformers()[0].GetPositions()    
    coordinates = torch.unsqueeze(torch.tensor(coordinates, dtype=torch.float, device=device), 0)
    
    atom_nums = []
    for atom in mol.GetAtoms():
        atom_nums.append(atom.GetAtomicNum())

    species = torch.tensor([atom_nums], device=device)
    total_energy = model((species, coordinates)).energies.item()
    
    return total_energy

def binding_energy(mol, model, device):
    atom_smiles = []
    for atom in mol.GetAtoms():
        atom_smile = atom.GetSymbol()
        atom_smiles.append(atom_smile)
    atom_mols = [Chem.MolFromSmiles(smile) for smile in atom_smiles]
    total_energies = hartree_energy(mol, model, device)
    atom_energies = sum([hartree_energy(mol, model, device) for mol in atom_mols])

    return total_energies - atom_energies
        
def main():
    args = read_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)


    data = pd.read_csv(args.data_path, header = None, names = ['smile', 'score'])
    
    energies = torch.empty(len(data))
    for i, smile in enumerate(data['smile']):
        mol = Chem.MolFromSmiles(smile)
        try:
            energy = binding_energy(mol, model, device)
        except Exception as e:
            print(e)
            energy = -1

        print(smile, energy)
        energies[i] = energy

    data['energies'] = energies.numpy()
    data.to_csv(args.data_path[:-4] + "_withenergies.csv", index = False)  

if __name__ == '__main__':
    main()
