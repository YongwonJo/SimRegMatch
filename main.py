import torch
from itertools import product

from utils.args import proposed_parser
from tasks.ProposedTrainer import ProposedTrainer

def main():        
    parser = proposed_parser()
    args = parser.parse_args()
    args.cuda = torch.device("cuda:0")
    
    trainer = ProposedTrainer(args)
    for epoch in range(1, args.epochs+1, 1):
        trainer.train(epoch)
        trainer.validation(epoch)
    
    trainer.inference(epoch)


if __name__ == "__main__":
    main()
