import struct_case as sc
import sme
import reader

def main():
    # Load solar system domain
    name1, facts1 = reader.read_meld_file('solar_system.meld')
    solar_system = sc.StructCase(facts1, name1)
    
    # Load atom domain
    name2, facts2 = reader.read_meld_file('atom.meld')
    atom = sc.StructCase(facts2, name2)

    print('Comparing Solar System vs Atom Structure:')
    print('=' * 50)
    
    # Compare them
    sme_engine = sme.SME(solar_system, atom)
    global_mappings = sme_engine.match()
    
    # Display results
    for i, mapping in enumerate(global_mappings, 1):
        print(f'\nMapping #{i}:')
        print(mapping)
        print(f'Similarity Score: {mapping.score:.4f}')
        print('-' * 50)

if __name__ == '__main__':
    main()
