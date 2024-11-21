import torch
from torch import optim
from utils.utils import *
from utils.tools import *
from models.score_models import *
from models.embed_models import *
from heterograph.geometric_dataset import *

def graph_embedding(args,data,graphs, conv, x_type,x_loss, emb_norm,out):
    model_init_seed = 2024
    if args.seed is not None:
        model_init_seed = args.seed

    set_random_seed(model_init_seed)
    num_nodes = data.x.size(0)
    num_features = data.x.size(1)
    num_classes = (data.y.max() + 1).item() if args.is_continuous == 'True' else None

    # model and train
    model = SVGA(
        data.edge_index, graphs, num_nodes, num_features, num_classes, args.hidden,
        args.Lambda, args.beta, args.layers, conv, args.dropout,
        x_type, x_loss, emb_norm, torch.nonzero(data.train_mask).squeeze(), data.x[data.train_mask],
        args.dec_bias, args.is_continuous)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(args.device)
    data = data.to(args.device)
    stopper = EarlyStopping(patience=args.patience)

    # report train log
    from datetime import datetime
    formatted_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    out_path = out + '/' + args.dataset + '/' + 'Embedding'
    saved_model_path = os.path.join(out_path, f'model_checkpoint_{formatted_date}.pth')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        print(f"Save folder already exists.")

    if not args.silent:
        print('-' * 47)
        print('epoch x_loss y_loss r_loss trn val')
    best_acc = np.inf if args.is_continuous else 0

    with open(out_path + '/' + 'trained result.txt', 'a', encoding='utf-8') as savefile:
        savefile.writelines('epoch \t x_loss \t y_loss \t r_loss \t trn \t val \n')
        savefile.close()
    for epoch in range(args.epochs + 1):
        loss_list = []
        for _ in range(args.updates):
            loss_list = update_model(step=(epoch > 0), model=model, edge_index=data.edge_index, data=data,
                                     graphs=graphs,
                                     x_features=data.x[data.train_mask], optimizer=optimizer,
                                     y_nodes=torch.nonzero(data.train_mask).squeeze(),
                                     y_labels=data.y[data.train_mask], is_continuous=args.is_continuous)
        trnloss = evaluate_model(model=model, edge_index=data.edge_index,
                                 x_nodes=torch.nonzero(data.train_mask).squeeze(),
                                 val_nodes=torch.nonzero(data.val_mask).squeeze(),
                                 is_continuous=args.is_continuous, x_all=data.x, data=data, graphs=graphs)
        curr_result = [epoch, loss_list, trnloss]

        val_loss = trnloss[1]
        if is_better(val_loss, best_acc, args.is_continuous):
            torch.save(model.state_dict(), saved_model_path)
            best_acc = val_loss

        if not args.silent:
            print_log(*curr_result)
        if args.save:
            with open(out_path + '/' + 'trained result.txt', 'a', encoding='utf-8') as savefile:
                savefile.writelines(
                    f'{epoch} \t {loss_list[0]} \t {loss_list[1]} \t {loss_list[2]} \t {trnloss[0]} \t {trnloss[1]} \n')
                savefile.close()

        early_stop = stopper.step(val_loss, model)
        if early_stop:
            break

    save_ebedding_res(model, saved_model_path, args, data, graphs, out_path)

    # plotting train results
    if args.plotting and args.save:
        plotting_train_res(out_path, 'trained result.txt', 'Embedding_Training_Plotting')

    print('Finished Graph embedding.')

    return out_path,formatted_date

def extracted_kn_step(g,out_path, data, molList, trans_adj_list, meta_paths,num_ntypes,args,formatted_date):
    train_mask = data.train_mask.to(args.device)
    num_classes = torch.sum(data.y) + 1 if not args.is_continuous else 1
    if args.embedding_step:
        feats = torch.load(os.path.join(out_path, 'features.pth')).to(args.device)
        labels = feats.sum(dim=1).to(args.device)
    else:
        feats = data.x.to(args.device)
        labels = data.y.to(args.device)

    if args.Heterogeneous:
        settings = []
        for mol in molList:
            settings.append({'T': 2, 'device': 2})
    else:
        settings = [{'T': 2, 'device': 2}]

    for i in range(len(trans_adj_list)):
        settings[i]['device'] = args.device
        settings[i]['TransM'] = trans_adj_list[i]

    # model
    model = HAN(meta_paths=meta_paths,
                in_size=feats.shape[1],
                hidden_size=args.hidden_size,
                out_size=num_classes,
                num_heads=args.num_heads,
                dropout=args.dropout,
                settings=settings,
                num_nodes=num_ntypes,
                is_continue=args.is_continuous,
                args=args).to(args.device)

    stopper = EarlyStopping(patience=args.patience)
    if args.is_continuous:
        loss_fcn = nn.MSELoss()
    else:
        if torch.sum(data.y) + 1 == 2:
            loss_fcn = nn.CrossEntropyLoss()
        else:
            loss_fcn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd)

    if not args.is_continuous:
        print('Epoch \t Train_loss \t Train_f1 \t Train_f2 \t Val_loss \t Val_f1 \t Val_f2 \n')
    else:
        print('Epoch \t Train_loss \t Train_mae \t Train_r2 \t Val_loss \t Val_mae \t Val_r2 \n')
    if args.save:
        if not args.is_continuous:
            with open(out_path + '/' + 'trained result_extracted key nodes.txt', 'a', encoding='utf-8') as savefile:
                savefile.writelines('epoch \t trn \t Train_f1 \t Train_f2 \t val \t Val_f1 \t Val_f2 \n')
                savefile.close()
        else:
            with open(out_path + '/' + 'trained result_extracted key nodes.txt', 'a', encoding='utf-8') as savefile:
                savefile.writelines('epoch \t trn \t Train_mae \t Train_r2 \t val \t Val_mae \t Val_r2 \n')
                savefile.close()

    for epoch in range(args.epochs):
        model.train()
        logits = model(g, feats)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_score1, train_score2, train_score3 = score(logits[train_mask], labels[train_mask], args.is_continuous)
        val_loss, val_score1, val_score2, val_score3 = evaluate(model, g, feats, labels, data.val_mask, loss_fcn,
                                                                args.is_continuous)
        early_stop = stopper.step(val_loss.data.item(), model)
        print('{} \t {} \t {} \t {} \t {} \t {} \t {} \n'.format(
            epoch, loss, train_score2, train_score3, val_loss, val_score2, val_score3
        ))
        if args.save:
            with open(out_path + '/' + 'trained result_extracted key nodes.txt', 'a', encoding='utf-8') as savefile:
                savefile.writelines(
                    f'{epoch} \t {loss} \t {train_score2} \t {train_score3} \t {val_loss} \t {val_score2} \t {val_score3} \n')
                savefile.close()

        if early_stop:
            break

    saved_exmodel_path = os.path.join(out_path, f'ex_model_checkpoint_{formatted_date}.pth')
    torch.save(model.state_dict(), saved_exmodel_path)
    with torch.no_grad():
        scores = model(g, feats)
        scores = scores + torch.abs(torch.min(scores))
    # plotting train results
    if args.plotting and args.save:
        plotting_train_res(out_path, 'trained result_extracted key nodes.txt', 'Extracting_Training_Plotting')

    return scores

