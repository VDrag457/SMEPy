#!/usr/bin/env python3
"""
PDDL Domain Similarity Calculator
Usage: python pddl_similarity.py <domain1.pddl> <domain2.pddl>
"""

import sys
import os
import struct_case as sc
import sme
import reader
from pddl_parser import PDDLParser
from pddl_to_meld import PDDLToMeld

def calculate_similarity(pddl_file1, pddl_file2):
    """Calculate similarity score between two PDDL domains."""
    
    # Check if files exist
    if not os.path.exists(pddl_file1):
        print(f"Error: File not found: {pddl_file1}")
        sys.exit(1)
    if not os.path.exists(pddl_file2):
        print(f"Error: File not found: {pddl_file2}")
        sys.exit(1)
    
    # Parse PDDL domains
    try:
        parser1 = PDDLParser(pddl_file1)
        parser2 = PDDLParser(pddl_file2)
    except Exception as e:
        print(f"Error parsing PDDL files: {e}")
        sys.exit(1)
    
    # Convert to .meld format (temporary files)
    converter1 = PDDLToMeld(parser1)
    converter2 = PDDLToMeld(parser2)
    
    temp_meld1 = '.temp_domain1.meld'
    temp_meld2 = '.temp_domain2.meld'
    
    try:
        converter1.write_meld_file(temp_meld1)
        converter2.write_meld_file(temp_meld2)
        
        # Load into SME
        name1, facts1 = reader.read_meld_file(temp_meld1)
        case1 = sc.StructCase(facts1, name1)
        
        name2, facts2 = reader.read_meld_file(temp_meld2)
        case2 = sc.StructCase(facts2, name2)
        
        # Run structure mapping
        sme_engine = sme.SME(case1, case2)
        global_mappings = sme_engine.match()
        
        return global_mappings
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_meld1):
            os.remove(temp_meld1)
        if os.path.exists(temp_meld2):
            os.remove(temp_meld2)

def main():
    if len(sys.argv) != 3:
        print("Usage: python pddl_similarity.py <domain1.pddl> <domain2.pddl>")
        print("\nExample:")
        print("  python pddl_similarity.py blocksworld.pddl logistics.pddl")
        sys.exit(1)
    
    pddl_file1 = sys.argv[1]
    pddl_file2 = sys.argv[2]
    
    print(f"Comparing: {os.path.basename(pddl_file1)} vs {os.path.basename(pddl_file2)}")
    print("-" * 60)
    
    mappings = calculate_similarity(pddl_file1, pddl_file2)
    
    if mappings:
        print(f"\nFound {len(mappings)} structural mapping(s):\n")
        for i, mapping in enumerate(mappings, 1):
            print(f"Mapping #{i}: Score = {mapping.score:.4f}")
        
        # Print best score
        best_score = max(m.score for m in mappings)
        print(f"\n{'='*60}")
        print(f"Best Similarity Score: {best_score:.4f}")
        print(f"{'='*60}")
    else:
        print("\nNo structural similarities found.")
        print("Best Similarity Score: 0.0000")

if __name__ == '__main__':
    main()
