import os
import argparse
import warnings



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str,
                        default='./example/4yhj.pdb')
    parser.add_argument('--lig_path', type=str,
                        default='./example/4yhj_reference_ligand_partial.sdf')
    parser.add_argument('--config', type=str, default='./configs/sample_for_pdb.yml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--outdir', type=str, default='./output')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed)
    
