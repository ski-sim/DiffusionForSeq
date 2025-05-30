

def get_dataset(args, oracle):
    if args.task == "amp":
        from lib.dataset.regression import AMPRegressionDataset
        return AMPRegressionDataset(args.proxy_data_split, args.num_folds, args, oracle)
    # elif args.task == "tfbind":
    #     from lib.dataset.regression import TFBind8Dataset
    #     return TFBind8Dataset(args, oracle)
    else:
        from lib.dataset.regression import BioSeqDataset
        return BioSeqDataset(args, oracle)
