# molecule_builder.py
"""
MoleculeBuilder provides BRICS-based molecule assembly utilities.

The class loads or receives a fragment library, identifies BRICS attachment
types, checks whether two attachment types can be connected, connects compatible
fragments, and builds complete molecules by iterative fragment assembly.
"""

import random
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Contrib.SA_Score import sascorer

class MoleculeBuilder:
    def __init__(self, fragment_list=None, fragment_file=None, max_steps=10):
        self.max_steps = max_steps
        if fragment_list is not None:
            self.all_fragments = fragment_list
        elif fragment_file is not None:
            self.all_fragments = self.load_fragments(fragment_file)
        else:
            raise ValueError("Either fragment_list or fragment_file must be provided.")
        self.frag_types = {frag: self.get_brics_attachment_types(frag) for frag in self.all_fragments}

    def load_fragments(self, fragment_file):
        df = pd.read_csv(fragment_file)
        return [s for s in df["SMILES"] if "*" in s]

    def brics_connection_allowed(self, type1, type2):
        allowed_pairs = {
            (1, 10), (1, 2), (1, 3), (1, 5), 
            (2, 1), (2, 12), (2, 14), (2, 16),
            (3, 1), (3, 13), (3, 14), (3, 16), (3, 4), (3, 15),
            (4, 3), (4, 11), (4, 5),
            (5, 4), (5, 13), (5, 15), (5, 16), (5, 1), (5, 12), (5, 14),
            (6, 13), (6, 14), (6, 15), (6, 16),
            (7, 7),
            (8, 9), (8, 10), (8, 13), (8, 14), (8, 15), (8, 16),
            (9, 8), (9, 15), (9, 16), (9, 14),
            (10, 1), (10, 14), (10, 8), (10, 13), (10, 16), (10, 15),
            (11, 15), (11, 4), (11, 16), (11, 13), (11, 14),
            (12, 2), (12, 5),
            (13, 8), (13, 14), (13, 11), (13, 10), (13, 5), (13, 15), (13, 6), (13, 16), (13, 3),
            (14, 8), (14, 11), (14, 10), (14, 15), (14, 6), (14, 16), (14, 3), (14, 2), (14, 13), (14, 14), (14, 5), (14, 9),
            (15, 5), (15, 10), (15, 11), (15, 14), (15, 8), (15, 13), (15, 9), (15, 3), (15, 16), (15, 6),
            (16, 3), (16, 9), (16, 2), (16, 8), (16, 14), (16, 11), (16, 10), (16, 15), (16, 6), (16, 5), (16, 16),
        }
        return (type1, type2) in allowed_pairs

    def find_attachment_idx_with_label(self, mol):
        points = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
                if neighbors:
                    label = atom.GetIsotope() if atom.GetIsotope() > 0 else None
                    points.append((atom.GetIdx(), neighbors[0], label))
        return points

    def connect_fragments(self, mol1, mol2):
        mol1 = Chem.RWMol(mol1)
        mol2 = Chem.RWMol(mol2)
        att1 = self.find_attachment_idx_with_label(mol1)
        att2 = self.find_attachment_idx_with_label(mol2)

        for a1_idx, n1_idx, l1 in att1:
            for a2_idx, n2_idx, l2 in att2:
                if self.brics_connection_allowed(l1, l2):
                    combined = Chem.CombineMols(mol1, mol2)
                    offset = mol1.GetNumAtoms()
                    emol = Chem.EditableMol(combined)

                    # Remove '*' atoms
                    for idx in sorted([a1_idx, a2_idx + offset], reverse=True):
                        emol.RemoveAtom(idx)

                    def adjust(idx, removed):
                        return idx - sum(1 for r in removed if r < idx)

                    n1 = adjust(n1_idx, [a1_idx, a2_idx + offset])
                    n2 = adjust(n2_idx + offset, [a1_idx, a2_idx + offset])

                    bond_order = Chem.rdchem.BondType.DOUBLE if (l1 == 7 and l2 == 7) else Chem.rdchem.BondType.SINGLE
                    emol.AddBond(n1, n2, order=bond_order)
                    try:
                        newmol = emol.GetMol()
                        Chem.SanitizeMol(newmol)
                        return newmol
                    except:
                        return None
        return None

    def has_attachment_points(self, mol):
        return any(atom.GetSymbol() == '*' for atom in mol.GetAtoms())

    def get_brics_attachment_types(self, smiles):
        return list(map(int, re.findall(r'\[(\d+)\*\]', smiles)))

    def can_connect_any(self, types1, types2):
        return any(self.brics_connection_allowed(t1, t2) for t1 in types1 for t2 in types2)

    def build_one_molecule(self, max_attempts=100):
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
    
            first_frag = random.choice(self.all_fragments)
            mol = Chem.MolFromSmiles(first_frag)
            if mol is None:
                continue
    
            steps = [Chem.MolToSmiles(mol)]
            used_fragments = [first_frag]
            step = 1
    
            while step < self.max_steps and self.has_attachment_points(mol):
                if not self.has_attachment_points(mol):
                    break
    
                current_types = self.get_brics_attachment_types(Chem.MolToSmiles(mol))
                candidates = [f for f, t in self.frag_types.items() if self.can_connect_any(current_types, t)]
                if not candidates:
                    break
    
                frag = random.choice(candidates)
                new_mol = self.connect_fragments(mol, Chem.MolFromSmiles(frag))
                if new_mol is None:
                    break
    
                mol = new_mol
                steps.append(Chem.MolToSmiles(mol))
                used_fragments.append(frag)
                step += 1
    
            # If attachment points remain, treat the attempt as failed and retry.
            if self.has_attachment_points(mol):
                continue
    
            try:
                Chem.SanitizeMol(mol)
                final_smiles = Chem.MolToSmiles(mol)
                sa = sascorer.calculateScore(mol)
                return {
                    "final_smiles": final_smiles,
                    "sa_score": sa,
                    "steps": steps,
                    "fragments": used_fragments
                }
            except:
                continue
    
        # Return None if all attempts fail.
        return None

    def build_batch(self, count=5):
        results = []
        while len(results) < count:
            res = self.build_one_molecule()
            if res:
                results.append(res)
        return results
    