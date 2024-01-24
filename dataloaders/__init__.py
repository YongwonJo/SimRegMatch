import os
import pandas as pd

from torch.utils.data import DataLoader
from dataloaders.datasets.AgeDB import AgeDB
from dataloaders.datasets.AgeDB_Unlabeled import AgeDB_Unlabeled

def make_semi_loader(args, num_workers=12):
    df = pd.read_csv(os.path.join(args.data_dir, f'{args.dataset}.csv'))
    
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    
    df_train = make_balanced_unlabeled(df_train, args)
    df_labeled, df_unlabeled = df_train[df_train['split_train']=='labeled'], df_train[df_train['split_train']=='unlabeled']
    df_labeled = make_reduced(df_labeled, args)
    df_labeled = df_labeled[df_labeled['split_train_reduced']=='use']

    labeled_set = AgeDB(data_dir=args.data_dir,
                        df = df_labeled,
                        img_size=args.img_size,
                        split='train'
                        )
    labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    unlabeled_set = AgeDB_Unlabeled(data_dir=args.data_dir,
                        df = df_unlabeled,
                        img_size=args.img_size,
                        split='train'
                        )
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    valid_set = AgeDB(data_dir=args.data_dir,
                        df = df_val,
                        img_size=args.img_size,
                        split='valid'
                        )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        
    test_set = AgeDB(data_dir=args.data_dir,
                        df = df_test,
                        img_size=args.img_size,
                        split='test'
                        )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return labeled_loader, unlabeled_loader, valid_loader, test_loader


def make_balanced_unlabeled(data, args):
    import random
    random.seed(args.seed)
    
    l_set, u_set = [], []
    for v in range(100):
        curr_df = data[data['age']==v]
        curr_data = curr_df['path'].values
        random.shuffle(curr_data)
        
        curr_size = len(curr_data) // 2
        l_set += list(curr_data[:curr_size])
        u_set += list(curr_data[curr_size:])
    
    print(f"Labeled Data: {len(l_set)} | Unlabeled Data: {len(u_set)}")
    
    assert len(set(l_set).intersection(set(u_set)))==0
    
    combined_set = dict(zip(l_set, ['labeled' for _ in range(len(l_set))]))
    combined_set.update(dict(zip(u_set, ['unlabeled' for _ in range(len(u_set))])))

    data['split_train'] = data['path'].map(combined_set)
    return data


def make_reduced(data, args):
    import random
    random.seed(args.seed)
    
    use_set, not_set = [], []
    for v in range(100):
        curr_df = data[data['age']==v]
        curr_data = curr_df['path'].values
        random.shuffle(curr_data)
        
        curr_size = int(len(curr_data) * args.labeled_ratio)
        use_set += list(curr_data[:curr_size])
        not_set += list(curr_data[curr_size:])
    
    print(f"Using Data: {len(use_set)} | Not using Data: {len(not_set)}")
    
    assert len(set(use_set).intersection(set(not_set)))==0
    
    combined_set = dict(zip(use_set, ['use' for _ in range(len(use_set))]))
    combined_set.update(dict(zip(not_set, ['not' for _ in range(len(not_set))])))

    data['split_train_reduced'] = data['path'].map(combined_set)

    return data
