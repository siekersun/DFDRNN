from main import parse, train, report
from src.model import DFDRNN

if __name__=="__main__":
    name = "Gdataset"
    nums = 7
    # nums = [1,2,3,4,5,6,7,8,9,10]
    # nums = [16,17,18,19,20]
    topks = 20

    args = parse(print_help=True)
    args.dataset_name = name
    if name=="lagcn":
        args.edge_dropout=0.5
        # args.embedding_dim=64
    else:
        args.edge_dropout=0.2
    args.drug_neighbor_num = 7
    args.disease_neighbor_num = 7
    train(args, DFDRNN)
    # report("runs")
