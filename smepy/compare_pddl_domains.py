import struct_case as sc
import sme
import reader
from pddl_parser import PDDLParser
from pddl_to_meld import PDDLToMeld

def main():
    print("PDDL Domain Similarity Analysis using Structure Mapping Engine")
    print("=" * 70)
    
    # Parse PDDL domains
    print("\n1. Parsing PDDL domains...")
    blocksworld_parser = PDDLParser('blocksworld.pddl')
    logistics_parser = PDDLParser('logistics.pddl')
    
    print(f"   - Blocksworld: {len(blocksworld_parser.actions)} actions, {len(blocksworld_parser.predicates)} predicates")
    print(f"   - Logistics: {len(logistics_parser.actions)} actions, {len(logistics_parser.predicates)} predicates")
    
    # Convert to .meld format
    print("\n2. Converting to SME format...")
    blocksworld_converter = PDDLToMeld(blocksworld_parser)
    blocksworld_converter.write_meld_file('blocksworld.meld')
    
    logistics_converter = PDDLToMeld(logistics_parser)
    logistics_converter.write_meld_file('logistics.meld')
    
    print("   - Created blocksworld.meld")
    print("   - Created logistics.meld")
    
    # Load into SME
    print("\n3. Loading domains into SME...")
    name1, facts1 = reader.read_meld_file('blocksworld.meld')
    blocksworld_case = sc.StructCase(facts1, name1)
    
    name2, facts2 = reader.read_meld_file('logistics.meld')
    logistics_case = sc.StructCase(facts2, name2)
    
    # Run structure mapping
    print("\n4. Computing structural mappings...")
    sme_engine = sme.SME(blocksworld_case, logistics_case)
    global_mappings = sme_engine.match()
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: Structural Similarities Found")
    print("=" * 70)
    
    if global_mappings:
        for i, mapping in enumerate(global_mappings, 1):
            print(f"\n--- Mapping #{i} ---")
            print(mapping)
            print(f"\nSimilarity Score: {mapping.score:.4f}")
            print("-" * 70)
    else:
        print("\nNo significant structural mappings found.")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("\nInterpretation:")
    print("- Higher scores indicate stronger structural similarity")
    print("- Mappings show corresponding actions/relations across domains")
    print("- SME finds similarities based on relational structure, not names")

if __name__ == '__main__':
    main()
