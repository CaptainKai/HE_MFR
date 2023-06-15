import trainner
import argparse

def main():
    parser = argparse.ArgumentParser(description='PyTorch amsoft training')
    parser.add_argument('--config_path', type=str, default='./cfgs/res_am_36_ddp_pipeline.py',
                        help='config path') # TOOD  SR_36_ddp_split_abla_pipeline
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    args = parser.parse_args()

    trainee = trainner.Trainer(args.local_rank, args.config_path)
    trainee.run()

if __name__ == '__main__':
    main()